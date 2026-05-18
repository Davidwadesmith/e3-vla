"""Generate teacher cache from real VLA rollouts.

Usage:
  # LIBERO spatial suite, 100 episodes
  uv run python scripts/generate_cache.py \
      env=libero_spatial \
      model=openpi \
      output_dir=/root/autodl-tmp/e3vla/teacher_cache \
      max_episodes=100

  # Single-task debug run
  uv run python scripts/generate_cache.py \
      env=libero_single \
      env.task=libero_spatial_pick_place \
      model=openpi \
      max_episodes=5
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e3vla.data.cache_writer import TeacherCacheWriter, CacheConfig
from e3vla.schema import Observation

logger = logging.getLogger(__name__)


def build_adapter(cfg: DictConfig):
    """Build VLA adapter from config. Supports OpenPI and generic model loading."""

    if cfg.model.provider == "openpi":
        try:
            from e3vla.adapters.openpi_adapter import OpenPIAdapter
            logger.info(f"Loading OpenPI model from {cfg.model.checkpoint}")
            return OpenPIAdapter(
                checkpoint_path=cfg.model.checkpoint,
                chunk_len=cfg.model.chunk_len,
                ae_layers=list(cfg.model.ae_layers),
            )
        except ImportError:
            logger.error(
                "OpenPI not installed. Install with: uv pip install -e '.[openpi]'\n"
                "Or use model.provider=generic and provide a checkpoint path."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load OpenPI model: {e}")
            raise

    elif cfg.model.provider == "generic":
        raise NotImplementedError(
            "Generic VLA adapter not yet implemented. "
            "Implement BaseVLAAdapter for your model in e3vla/adapters/."
        )
    else:
        raise ValueError(f"Unknown model provider: {cfg.model.provider}")


def build_env(cfg: DictConfig):
    """Build LIBERO environment from config."""
    try:
        import libero
    except ImportError:
        logger.error(
            "LIBERO not installed. Install with:\n"
            "  pip install libero\n"
            "See https://github.com/rail-berkeley/LIBERO for details."
        )
        raise

    task_suite = cfg.env.suite  # e.g. "libero_spatial", "libero_object", "libero_goal"
    task_name = cfg.env.get("task", None)  # None = all tasks in suite

    # Get tasks
    if task_name:
        tasks = [task_name]
    else:
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_obj = benchmark_dict[task_suite]()
        tasks = list(range(task_suite_obj.n_tasks))

    logger.info(f"Task suite: {task_suite}, {len(tasks)} tasks")

    return _LIBEROEnvFactory(task_suite, tasks)


class _LIBEROEnvFactory:
    """Factory that creates LIBERO environments on demand."""

    def __init__(self, suite: str, task_indices: list):
        self.suite = suite
        self.task_indices = task_indices

    def __len__(self):
        return len(self.task_indices)

    def create(self, task_idx: int, seed: Optional[int] = None):
        import libero
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.suite]()

        task = task_suite.get_task(self.task_indices[task_idx % len(self.task_indices)])
        env_args = {
            "task": task,
            "bddl_file": task.bddl_file,
            "camera_heights": 224,
            "camera_widths": 224,
        }
        if seed is not None:
            env_args["seed"] = seed

        env = task_suite.get_env(**env_args)
        return env, task


def run_rollout(
    adapter,
    env_factory,
    task_idx: int,
    seed: int,
    cache_writer: TeacherCacheWriter,
    max_steps: int = 300,
) -> int:
    """Run a single episode rollout and write timesteps to cache.

    Returns number of timesteps collected.
    """
    env, task = env_factory.create(task_idx, seed=seed)
    task_name = getattr(task, "name", str(task_idx))
    task_id = getattr(task, "task_name", task_name)

    cache_writer.start_episode(
        episode_id=f"{task_name}_{seed}",
        task_id=task_id,
        instruction=getattr(task, "language", ""),
    )

    obs = env.reset()
    step = 0
    done = False
    n_collected = 0

    while not done and step < max_steps:
        wrapped_obs = Observation(
            image=torch.tensor(obs["image"]),
            instruction=getattr(task, "language", ""),
            robot_state=torch.tensor(obs["robot_state"]),
        )

        with torch.no_grad():
            action_chunk = adapter.full_inference(wrapped_obs)

        # Write to teacher cache
        cache_writer.add_timestep(wrapped_obs, action_chunk)
        n_collected += 1

        # Execute actions in environment
        actions_np = action_chunk.detach().cpu().numpy()
        for i in range(min(len(actions_np), cache_writer.config.chunk_len)):
            obs, reward, done, info = env.step(actions_np[i])
            step += 1
            if done:
                break

    cache_writer.end_episode()
    env.close()

    success = info.get("success", False) if "info" in dir() else False
    logger.info(f"  Episode {task_name}_{seed}: {step} steps, "
                f"success={success}, collected={n_collected}")

    return n_collected


@hydra.main(version_base=None, config_path="../configs", config_name="cache/default")
def main(cfg: DictConfig):
    logger.info("=== E3-VLA Teacher Cache Generation ===")
    logger.info(OmegaConf.to_yaml(cfg))

    # 1. Build adapter (real VLA model)
    adapter = build_adapter(cfg).to(cfg.device)
    adapter.eval()

    # 2. Build environments
    env_factory = build_env(cfg)

    # 3. Create cache writer
    cache_config = CacheConfig(
        cache_dir=cfg.output_dir,
        shard_size=cfg.shard_size,
        chunk_len=cfg.model.chunk_len,
        max_episodes=cfg.max_episodes,
    )
    writer = TeacherCacheWriter(cache_config, adapter)

    # 4. Run rollouts
    n_episodes = min(cfg.max_episodes, len(env_factory) * cfg.episodes_per_task)
    n_total_timesteps = 0
    t0 = time.time()

    logger.info(f"Starting {n_episodes} episodes...")
    log_interval = max(1, n_episodes // 20)  # progress every 5%

    for ep in range(n_episodes):
        task_idx = ep % len(env_factory)
        seed = cfg.seed_base + ep

        try:
            n_steps = run_rollout(
                adapter, env_factory, task_idx, seed,
                writer, max_steps=cfg.max_steps_per_episode,
            )
            n_total_timesteps += n_steps
            if ep % log_interval == 0 or ep == n_episodes - 1:
                logger.info(f"  [{ep+1}/{n_episodes}] {n_total_timesteps} timesteps collected "
                            f"({(ep+1)/n_episodes*100:.0f}%)")
        except Exception as e:
            logger.error(f"Episode {ep} (task {task_idx}) failed: {e}")
            continue

    # 5. Finalize
    writer.finalize()
    elapsed = time.time() - t0

    logger.info(f"=== Complete ===")
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Total timesteps: {n_total_timesteps}")
    logger.info(f"Time: {elapsed / 60:.1f} min")
    logger.info(f"Output: {cfg.output_dir}")
    logger.info(f"Records: {writer.num_records}")

    # Write summary
    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "n_episodes": n_episodes,
        "n_timesteps": n_total_timesteps,
        "elapsed_minutes": elapsed / 60,
        "model_provider": cfg.model.provider,
        "output_dir": cfg.output_dir,
    }
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
