"""Run full benchmark comparison across all methods.

Usage:
  uv run python scripts/run_benchmark.py \
      benchmark=ablation \
      checkpoint_dir=/root/autodl-tmp/e3vla/checkpoints
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Configure verbose logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e3vla.schema import ActionCommand
from e3vla.eval.rollout_runner import RolloutRunner
from e3vla.eval.report_writer import ReportWriter, MethodTaskResult


def build_libero_env_factory(cfg: DictConfig):
    """Create LIBERO environment factory."""
    try:
        import libero
        from libero.libero import benchmark
    except ImportError:
        logger.error("LIBERO not installed. pip install libero")
        raise

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.env.suite]()

    def factory(task_name, seed=None):
        task = task_suite.get_task_by_name(task_name)
        env = task_suite.get_env(task=task, seed=seed)
        return env

    return factory, list(task_suite.get_task_names())


def build_policy(cfg: DictConfig, checkpoint_dir: str, method: str):
    """Build a policy instance for a given method name."""
    from e3vla.drafter.cached_ae_drafter import CachedAEDrafter
    from e3vla.drafter.ablation.no_cached_ae import NoCachedAEDrafter
    from e3vla.drafter.ablation.cached_ae_no_offset import CachedAENoOffsetDrafter
    from e3vla.policy.speculative_cached_policy import SpeculativeCachedPolicy
    from e3vla.verifier.action_expert_anchor_verifier import ActionExpertAnchorVerifier
    from e3vla.verifier.prefix_acceptor import PrefixAcceptor

    method_configs = {
        "no_cached_ae": {
            "drafter_class": NoCachedAEDrafter,
            "drafter_kwargs": dict(hidden_dim=512, action_dim=7, chunk_len=16,
                                    num_layers=2, num_heads=8, D_r=16),
            "ckpt_name": "no_cached_ae",
        },
        "cached_ae_no_offset": {
            "drafter_class": CachedAENoOffsetDrafter,
            "drafter_kwargs": dict(hidden_dim=512, action_dim=7, chunk_len=16,
                                    num_layers=2, num_heads=8, D_r=16),
            "ckpt_name": "no_offset",
        },
        "ours": {
            "drafter_class": CachedAEDrafter,
            "drafter_kwargs": dict(hidden_dim=512, action_dim=7, chunk_len=16,
                                    num_layers=2, num_heads=8, D_r=16, D_ee=7),
            "ckpt_name": "full_offset",
        },
    }

    if method not in method_configs:
        # Try to load as a wrapper
        return _build_wrapper_policy(method, cfg, checkpoint_dir)

    mc = method_configs[method]

    # Load drafter from checkpoint
    drafter = mc["drafter_class"](**mc["drafter_kwargs"])
    ckpt_path = os.path.join(checkpoint_dir, mc["ckpt_name"], "best.pt")

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        drafter.load_state_dict(
            state.get("model_state_dict", state), strict=False
        )
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"Checkpoint not found: {ckpt_path}. Using random weights.")

    drafter = drafter.cuda().eval()

    # Build adapter (lazy import)
    from e3vla.adapters.openpi_adapter import OpenPIAdapter
    adapter = OpenPIAdapter(
        checkpoint_path=cfg.model.checkpoint,
        chunk_len=cfg.model.chunk_len,
    )

    # Build verifier + acceptor
    verifier = ActionExpertAnchorVerifier(
        t_list=tuple(cfg.verifier.t_list),
        tau_radius=cfg.verifier.tau_radius,
        dist_dims=cfg.verifier.dist_dims,
        eval_h=cfg.verifier.eval_h,
    )
    acceptor = PrefixAcceptor(
        tau_pos=cfg.acceptor.tau_pos,
        tau_rot=cfg.acceptor.tau_rot,
        tau_grip=cfg.acceptor.tau_grip,
        alpha_uncert=cfg.acceptor.alpha_uncert,
    )

    # Build policy
    policy = SpeculativeCachedPolicy(
        adapter=adapter,
        drafter=drafter,
        verifier=verifier,
        prefix_acceptor=acceptor,
        chunk_len=cfg.model.chunk_len,
        max_cache_age=cfg.policy.max_cache_age,
        periodic_full_every_n=cfg.policy.periodic_full_every_n,
        full_exec_len=cfg.policy.full_exec_len,
    )

    # Attach metadata
    policy.method_name = _get_method_name(method)
    policy.method_type = _get_method_type(method)

    return policy


def _get_method_name(method: str) -> str:
    names = {
        "no_cached_ae": "NoCachedAE (FLASH baseline)",
        "cached_ae_no_offset": "CachedAE-NoOffset",
        "ours": "CachedAE-FullOffset (Ours)",
        "full_vla": "Full VLA (π₀)",
        "action_reuse": "Cached Full Action Reuse",
        "action_reuse_offset": "Cached Action Reuse + Offset",
        "flash": "FLASH-style Drafter + Verifier",
    }
    return names.get(method, method)


def _get_method_type(method: str) -> str:
    types = {
        "full_vla": "full_vla",
        "no_cached_ae": "drafter_verify",
        "cached_ae_no_offset": "cached_drafter",
        "ours": "cached_drafter",
        "action_reuse": "action_reuse",
        "action_reuse_offset": "action_reuse",
        "flash": "drafter_verify",
    }
    return types.get(method, "unknown")


def _build_wrapper_policy(method: str, cfg: DictConfig, checkpoint_dir: str):
    """Build wrapper policies (Full VLA, FLASH, Action Reuse)."""
    if method == "full_vla":
        from e3vla.adapters.openpi_adapter import OpenPIAdapter

        class FullVLAPolicy:
            def __init__(self, adapter):
                self.adapter = adapter

            def reset(self, task_info=None):
                pass

            def act(self, obs):
                chunk = self.adapter.full_inference(obs)
                return ActionCommand(
                    actions=chunk[:8], execute_len=8,
                    can_interrupt=True, mode="full_refresh",
                )

            @property
            def method_name(self):
                return "Full VLA (π₀)"

            @property
            def method_type(self):
                return "full_vla"

            def get_metrics(self):
                return {}

            def get_diagnostics(self):
                return {}

        adapter = OpenPIAdapter(checkpoint_path=cfg.model.checkpoint)
        return FullVLAPolicy(adapter)

    elif method == "action_reuse":
        from e3vla.benchmark.wrappers.cached_action_reuse_wrapper import (
            CachedFullActionReuseWrapper,
        )
        return CachedFullActionReuseWrapper(dict(cfg.policy))

    elif method == "action_reuse_offset":
        from e3vla.benchmark.wrappers.cached_action_reuse_wrapper import (
            CachedActionReuseOffsetWrapper,
        )
        return CachedActionReuseOffsetWrapper(dict(cfg.policy))

    elif method == "flash":
        try:
            from e3vla.benchmark.wrappers.flash_wrapper import FLASHWrapper
            return FLASHWrapper({
                "base_policy_path": cfg.model.checkpoint,
                "draft_ckpt_path": os.path.join(checkpoint_dir, "flash_draft.pt"),
                "chunk_len": cfg.model.chunk_len,
            })
        except ImportError:
            logger.error("FLASH not installed. uv pip install -e '.[flash]'")
            raise

    raise ValueError(f"Unknown method: {method}")


def run_single_method(
    method: str,
    cfg: DictConfig,
    checkpoint_dir: str,
    env_factory,
    task_names: List[str],
    output_dir: str,
) -> List[MethodTaskResult]:
    """Run benchmark for a single method across all tasks."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Method: {_get_method_name(method)}")
    logger.info(f"{'='*60}")

    policy = build_policy(cfg, checkpoint_dir, method)

    runner = RolloutRunner(
        env_factory=env_factory,
        max_steps=cfg.max_steps_per_episode,
        action_horizon=cfg.model.chunk_len,
    )

    results = []
    n_episodes = cfg.num_episodes
    log_interval = max(1, n_episodes // 10)  # progress every 10%

    for task_name in task_names:
        logger.info(f"  Task: {task_name} ({n_episodes} episodes)")
        ep_results = []
        t_task_start = time.time()

        for ep in range(n_episodes):
            seed = cfg.seeds[ep % len(cfg.seeds)]
            result = runner.run_episode(policy, task_name, seed=seed)
            ep_results.append(result)

            if ep % log_interval == 0 or ep == n_episodes - 1:
                succ_so_far = sum(1 for r in ep_results if r.success)
                avg_lat = sum(r.average_latency_ms for r in ep_results) / len(ep_results)
                logger.info(
                    f"    [{ep+1}/{n_episodes}] success={succ_so_far}/{ep+1} "
                    f"({succ_so_far/(ep+1)*100:.0f}%) | avg_latency={avg_lat:.1f}ms"
                )

        elapsed_task = time.time() - t_task_start

        # Aggregate
        successes = sum(1 for r in ep_results if r.success)
        avg_steps = sum(r.total_steps for r in ep_results) / len(ep_results)
        avg_latency = sum(r.average_latency_ms for r in ep_results) / len(ep_results)
        avg_prefix = sum(r.avg_accepted_prefix for r in ep_results) / len(ep_results)
        total_fallback = sum(r.fallback_count for r in ep_results)
        total_spec = sum(r.speculative_count for r in ep_results)
        total_full = sum(r.full_refresh_count for r in ep_results)
        total_steps_all = sum(r.total_steps for r in ep_results)

        fallback_rate = total_fallback / max(1, total_fallback + total_spec)
        full_refresh_rate = total_full / max(1, total_steps_all)

        logger.info(
            f"    Task complete in {elapsed_task:.1f}s: success={successes}/{n_episodes} "
            f"({successes/n_episodes:.1%}) | latency={avg_latency:.1f}ms | "
            f"prefix={avg_prefix:.1f} | fallback={fallback_rate:.1%}"
        )

        results.append(MethodTaskResult(
            method=_get_method_name(method),
            method_type=_get_method_type(method),
            task=task_name,
            num_episodes=len(ep_results),
            success_rate=successes / len(ep_results),
            avg_steps=avg_steps,
            avg_latency_ms=avg_latency,
            avg_accepted_prefix=avg_prefix,
            fallback_rate=fallback_rate,
            full_refresh_rate=full_refresh_rate,
            speedup_vs_full=1.0,
        ))

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark/default")
def main(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("E3-VLA Benchmark Evaluation")
    logger.info("=" * 60)
    logger.info(f"Methods: {cfg.methods}")
    logger.info(f"Checkpoints: {cfg.checkpoint_dir}")
    logger.info(f"Model: {cfg.model.checkpoint}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_total:.1f} GB)")

    # Build env
    env_factory, task_names = build_libero_env_factory(cfg)

    if cfg.get("max_tasks", 0) > 0:
        task_names = task_names[:cfg.max_tasks]

    logger.info(f"Tasks: {len(task_names)} tasks — {task_names}")
    logger.info(f"Episodes per task: {cfg.num_episodes}")
    logger.info(f"Seeds: {len(cfg.seeds)} seeds")
    logger.info(f"Total episodes: {len(task_names) * cfg.num_episodes} per method × "
                f"{len(cfg.methods)} methods = "
                f"{len(task_names) * cfg.num_episodes * len(cfg.methods)} total")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Run each method
    report = ReportWriter(baseline_method="Full VLA (π₀)")
    all_results = []
    t_total_start = time.time()

    for method_idx, method in enumerate(cfg.methods):
        logger.info(f"\n{'='*60}")
        logger.info(f"Method {method_idx+1}/{len(cfg.methods)}: {_get_method_name(method)}")
        logger.info(f"{'='*60}")
        t_method_start = time.time()

        try:
            results = run_single_method(
                method, cfg, cfg.checkpoint_dir,
                env_factory, task_names, cfg.output_dir,
            )
            for r in results:
                report.add_result(r)
            all_results.extend(results)

            elapsed_method = time.time() - t_method_start
            logger.info(f"Method complete in {elapsed_method/60:.1f} min")
        except Exception as e:
            logger.error(f"Method '{method}' FAILED: {e}")
            logger.exception(e)
            continue

    elapsed_total = time.time() - t_total_start
    logger.info(f"\nAll methods complete in {elapsed_total/60:.1f} min")

    # Compute speedup vs baseline
    baseline_latency = {}
    for r in all_results:
        if r.method == "Full VLA (π₀)":
            baseline_latency[r.task] = r.avg_latency_ms

    for r in all_results:
        baseline = baseline_latency.get(r.task, r.avg_latency_ms)
        r.speedup_vs_full = baseline / max(r.avg_latency_ms, 1e-6)

    # Save
    results_path = os.path.join(cfg.output_dir, "benchmark_results.json")
    report.to_json(results_path)
    logger.info(f"Results saved: {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info("\n" + report.main_table())
    logger.info("\n" + report.latency_breakdown_table())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
