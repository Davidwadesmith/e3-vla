"""Benchmark runner — shared evaluation infrastructure.

All methods are evaluated through the same BenchmarkRunner
to ensure fair comparison.
"""

from typing import List, Dict, Any
import time
import json

import torch

from e3vla.schema import ActionCommand, PolicyMetrics


class LatencyProfiler:
    """Per-component latency measurement."""

    def __init__(self):
        self._records: List[Dict[str, float]] = []

    def record(self, **kwargs) -> None:
        self._records.append(kwargs)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Compute p50/p90/p95/mean/std for each recorded key."""
        if not self._records:
            return {}

        keys = self._records[0].keys()
        result = {}
        for key in keys:
            vals = sorted(r[key] for r in self._records if key in r)
            if not vals:
                continue
            n = len(vals)
            result[key] = {
                "p50": vals[n // 2],
                "p90": vals[int(n * 0.9)],
                "p95": vals[int(n * 0.95)],
                "mean": sum(vals) / n,
                "std": (sum((v - sum(vals)/n)**2 for v in vals) / n) ** 0.5,
            }
        return result


class MetricsCollector:
    """Collects success rate, latency, fallback, accepted prefix stats."""

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []

    def record_episode(
        self,
        success: bool,
        total_steps: int,
        total_latency: float,
        fallback_count: int,
        total_accepted_prefix: int,
        speculative_count: int,
        policy_metrics: dict,
    ) -> None:
        self.episodes.append({
            "success": success,
            "total_steps": total_steps,
            "total_latency": total_latency,
            "fallback_count": fallback_count,
            "total_accepted_prefix": total_accepted_prefix,
            "speculative_count": speculative_count,
            **policy_metrics,
        })

    def summary(self) -> Dict[str, float]:
        """Aggregate over all episodes."""
        if not self.episodes:
            return {}

        n = len(self.episodes)
        successes = sum(1 for e in self.episodes if e["success"])
        total_steps = sum(e["total_steps"] for e in self.episodes)
        total_latency = sum(e["total_latency"] for e in self.episodes)
        total_fallbacks = sum(e["fallback_count"] for e in self.episodes)
        total_accepted = sum(e["total_accepted_prefix"] for e in self.episodes)
        total_spec = sum(e["speculative_count"] for e in self.episodes)

        return {
            "success_rate": successes / n,
            "avg_steps_per_episode": total_steps / n,
            "avg_latency_per_step": total_latency / max(1, total_steps),
            "fallback_rate": total_fallbacks / max(1, total_fallbacks + total_spec),
            "avg_accepted_prefix": total_accepted / max(1, total_spec),
            "num_episodes": n,
        }


class BenchmarkRunner:
    """Shared evaluation infrastructure for all methods."""

    def __init__(
        self,
        env_factory,
        max_steps: int = 300,
        num_episodes: int = 50,
        seeds: list = None,
    ):
        self.env_factory = env_factory
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.seeds = seeds or [42]
        self.profiler = LatencyProfiler()

    def run(self, policy, task_spec) -> MetricsCollector:
        """Run a single (policy, task) evaluation."""
        metrics = MetricsCollector()

        for ep in range(self.num_episodes):
            seed = self.seeds[ep % len(self.seeds)]
            result = self._run_episode(policy, task_spec, seed)
            metrics.record_episode(**result)

        return metrics

    def _run_episode(self, policy, task_spec, seed: int) -> dict:
        """Run a single episode."""
        env = self.env_factory(task_spec, seed=seed)
        policy.reset(task_spec)

        obs_raw = env.reset()
        obs = self._wrap_obs(obs_raw)

        done = False
        step = 0
        total_latency = 0.0
        fallback_count = 0
        total_accepted = 0
        spec_count = 0
        success = False

        while not done and step < self.max_steps:
            t0 = time.perf_counter()
            action_cmd = policy.act(obs)
            elapsed = time.perf_counter() - t0

            total_latency += elapsed
            self.profiler.record(act_latency=elapsed * 1000)

            if action_cmd.mode == "fallback":
                fallback_count += 1
            elif action_cmd.mode == "speculative":
                spec_count += 1
                total_accepted += action_cmd.prefix_length

            obs_raw, reward, done, info = env.step(
                action_cmd.actions.numpy()
            )
            obs = self._wrap_obs(obs_raw)
            step += 1

            if info.get("success", False):
                success = True
                done = True

        env.close()
        policy_metrics = {}
        if hasattr(policy, 'get_metrics'):
            policy_metrics = policy.get_metrics()

        return {
            "success": success,
            "total_steps": step,
            "total_latency": total_latency,
            "fallback_count": fallback_count,
            "total_accepted_prefix": total_accepted,
            "speculative_count": spec_count,
            "policy_metrics": policy_metrics,
        }

    @staticmethod
    def _wrap_obs(obs_raw) -> "Observation":
        """Wrap raw env observation into E3-VLA Observation."""
        from e3vla.schema import Observation
        return Observation(
            image=torch.tensor(obs_raw.get("image", [])),
            instruction=obs_raw.get("instruction", ""),
            robot_state=torch.tensor(obs_raw.get("robot_state", [])),
        )
