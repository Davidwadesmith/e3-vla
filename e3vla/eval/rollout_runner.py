"""Rollout runner for evaluating policies in environments."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch

from e3vla.schema import ActionCommand


@dataclass
class EpisodeResult:
    success: bool
    total_steps: int
    total_wall_time: float
    average_latency_ms: float
    fallback_count: int
    full_refresh_count: int
    speculative_count: int
    avg_accepted_prefix: float
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    logs: list = field(default_factory=list)


class RolloutRunner:
    """Executes a policy in an environment for a fixed number of episodes.

    Collects standardized EpisodeResult for the benchmark pipeline.
    """

    def __init__(
        self,
        env_factory,
        max_steps: int = 300,
        action_horizon: int = 16,
    ):
        self.env_factory = env_factory
        self.max_steps = max_steps
        self.action_horizon = action_horizon

    def run_episode(
        self, policy, task_spec, seed: Optional[int] = None
    ) -> EpisodeResult:
        env = self.env_factory(task_spec, seed=seed)
        if hasattr(policy, 'reset'):
            policy.reset(task_spec)

        obs_raw = env.reset()
        obs = self._wrap_obs(obs_raw)

        done = False
        step = 0
        total_latency = 0.0
        fallback_count = 0
        full_refresh_count = 0
        spec_count = 0
        total_accepted = 0
        latency_records = []
        success = False

        while not done and step < self.max_steps:
            t0 = time.perf_counter()
            action_cmd = policy.act(obs)
            elapsed = time.perf_counter() - t0
            latency_ms = elapsed * 1000.0

            total_latency += elapsed
            latency_records.append({
                "step": step,
                "latency_ms": latency_ms,
                "mode": action_cmd.mode,
            })

            if action_cmd.mode == "full_refresh":
                full_refresh_count += 1
            elif action_cmd.mode == "fallback":
                fallback_count += 1
            elif action_cmd.mode == "speculative":
                spec_count += 1
                total_accepted += action_cmd.prefix_length

            # Step environment
            actions_np = action_cmd.actions.detach().cpu().numpy()
            if actions_np.ndim == 3:
                actions_np = actions_np.squeeze(0)

            for i in range(action_cmd.execute_len):
                if i >= len(actions_np):
                    break
                obs_raw, reward, done, info = env.step(actions_np[i])
                step += 1
                if done:
                    break

            obs = self._wrap_obs(obs_raw)

            if info.get("success", False):
                success = True
                done = True

        env.close()

        avg_latency = total_latency / max(step, 1)
        avg_prefix = total_accepted / max(spec_count, 1)

        return EpisodeResult(
            success=success,
            total_steps=step,
            total_wall_time=total_latency,
            average_latency_ms=avg_latency * 1000,
            fallback_count=fallback_count,
            full_refresh_count=full_refresh_count,
            speculative_count=spec_count,
            avg_accepted_prefix=avg_prefix,
            latency_breakdown={},
            logs=latency_records,
        )

    @staticmethod
    def _wrap_obs(obs_raw):
        from e3vla.schema import Observation
        image = obs_raw.get("image")
        if image is not None and not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        robot_state = obs_raw.get("robot_state")
        if robot_state is not None and not isinstance(robot_state, torch.Tensor):
            robot_state = torch.tensor(robot_state)
        return Observation(
            image=image or torch.zeros(1, 3, 224, 224),
            instruction=obs_raw.get("instruction", ""),
            robot_state=robot_state or torch.zeros(16),
        )
