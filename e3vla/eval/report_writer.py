"""Report writer: aggregates benchmark results into tables and plots."""

import json
from typing import List, Dict, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MethodTaskResult:
    method: str
    method_type: str
    task: str
    num_episodes: int
    success_rate: float
    avg_steps: float
    avg_latency_ms: float
    avg_accepted_prefix: float
    fallback_rate: float
    full_refresh_rate: float
    speedup_vs_full: float = 1.0
    latency_breakdown: Dict[str, float] = field(default_factory=dict)


class ReportWriter:
    """Aggregates episode results into paper-ready tables."""

    def __init__(self, baseline_method: str = "Full VLA"):
        self._results: List[MethodTaskResult] = []
        self._baseline = baseline_method
        self._baseline_latency: Dict[str, float] = {}

    def add_result(self, result: MethodTaskResult) -> None:
        self._results.append(result)
        if result.method == self._baseline:
            self._baseline_latency[result.task] = result.avg_latency_ms

    def main_table(self) -> str:
        """Generate main comparison table."""
        lines = [
            "| Method | Success ↑ | E2E Latency (ms) ↓ | Speedup ↑ | Accepted Prefix ↑ | Fallback Rate ↓ |",
            "|--------|-----------|--------------------|-----------|-------------------|-----------------|",
        ]
        for r in sorted(self._results, key=lambda x: x.success_rate, reverse=True):
            # Compute speedup relative to baseline
            baseline = self._baseline_latency.get(r.task, r.avg_latency_ms)
            speedup = baseline / max(r.avg_latency_ms, 1e-6)

            lines.append(
                f"| {r.method} | {r.success_rate:.1%} | {r.avg_latency_ms:.1f} | "
                f"{speedup:.2f}× | {r.avg_accepted_prefix:.1f} | {r.fallback_rate:.1%} |"
            )
        return "\n".join(lines)

    def latency_breakdown_table(self) -> str:
        """Generate latency breakdown table."""
        lines = [
            "| Method | Refresh | Cache Read | Offset Align | Draft | Verify | Total |",
            "|--------|---------|------------|-------------|-------|--------|-------|",
        ]
        for r in self._results:
            bd = r.latency_breakdown
            lines.append(
                f"| {r.method} | {bd.get('refresh', 0):.1f} | "
                f"{bd.get('cache_read', 0):.1f} | {bd.get('offset_align', 0):.1f} | "
                f"{bd.get('draft', 0):.1f} | {bd.get('verify', 0):.1f} | "
                f"{r.avg_latency_ms:.1f} |"
            )
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        data = {
            "results": [
                {
                    "method": r.method,
                    "method_type": r.method_type,
                    "task": r.task,
                    "success_rate": r.success_rate,
                    "avg_latency_ms": r.avg_latency_ms,
                    "speedup_vs_full": r.speedup_vs_full,
                    "avg_accepted_prefix": r.avg_accepted_prefix,
                    "fallback_rate": r.fallback_rate,
                    "full_refresh_rate": r.full_refresh_rate,
                    "latency_breakdown": r.latency_breakdown,
                }
                for r in self._results
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def results(self) -> List[MethodTaskResult]:
        return self._results
