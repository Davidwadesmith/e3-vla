"""Generate paper-ready tables and plots from benchmark results.

Usage:
  uv run python scripts/generate_report.py \
      --results experiments/results/benchmark_results.json \
      --output experiments/report/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e3vla.eval.report_writer import MethodTaskResult, ReportWriter


def load_results(path: str) -> List[MethodTaskResult]:
    """Load benchmark results from JSON."""
    with open(path) as f:
        data = json.load(f)

    results = []
    for r in data.get("results", data if isinstance(data, list) else []):
        results.append(MethodTaskResult(**r))
    return results


def generate_main_csv(results: List[MethodTaskResult], output_dir: str):
    """Generate main comparison CSV."""
    if not results:
        return

    # Aggregate by method across tasks
    methods = {}
    for r in results:
        if r.method not in methods:
            methods[r.method] = []
        methods[r.method].append(r)

    rows = []
    for method, method_results in methods.items():
        avg_success = np.mean([r.success_rate for r in method_results])
        avg_latency = np.mean([r.avg_latency_ms for r in method_results])
        avg_speedup = np.mean([r.speedup_vs_full for r in method_results])
        avg_prefix = np.mean([r.avg_accepted_prefix for r in method_results])
        avg_fallback = np.mean([r.fallback_rate for r in method_results])

        rows.append({
            "Method": method,
            "Success": f"{avg_success:.4f}",
            "E2E Latency (ms)": f"{avg_latency:.1f}",
            "Speedup vs Full VLA": f"{avg_speedup:.2f}x",
            "Accepted Prefix": f"{avg_prefix:.1f}",
            "Fallback Rate": f"{avg_fallback:.4f}",
        })

    # Sort by success rate
    rows.sort(key=lambda r: float(r["Success"]), reverse=True)

    path = os.path.join(output_dir, "main_table.csv")
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Main table: {path}")


def generate_main_markdown(results: List[MethodTaskResult], output_dir: str):
    """Generate paper-ready markdown table."""
    report = ReportWriter()
    for r in results:
        report.add_result(r)

    table = report.main_table()
    path = os.path.join(output_dir, "main_table.md")
    with open(path, "w") as f:
        f.write(table)
    print(f"Markdown table: {path}")


def generate_latency_breakdown_csv(results: List[MethodTaskResult], output_dir: str):
    """Generate latency breakdown CSV."""
    rows = []
    for r in results:
        bd = r.latency_breakdown
        rows.append({
            "Method": r.method,
            "Task": r.task,
            "Refresh (ms)": f"{bd.get('refresh', 0):.1f}",
            "Cache Read (ms)": f"{bd.get('cache_read', 0):.1f}",
            "Offset Align (ms)": f"{bd.get('offset_align', 0):.1f}",
            "Draft (ms)": f"{bd.get('draft', 0):.1f}",
            "Verify (ms)": f"{bd.get('verify', 0):.1f}",
            "Total (ms)": f"{r.avg_latency_ms:.1f}",
        })

    if not rows:
        return

    path = os.path.join(output_dir, "latency_breakdown.csv")
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Latency breakdown: {path}")


def generate_metrics_json(results: List[MethodTaskResult], output_dir: str):
    """Dump all metrics as structured JSON."""
    data = {
        "methods": [
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
            for r in results
        ],
        "aggregated": {},
    }

    # Per-method aggregation
    methods = {}
    for r in results:
        if r.method not in methods:
            methods[r.method] = {"success": [], "latency": [], "speedup": [], "prefix": [], "fallback": []}
        m = methods[r.method]
        m["success"].append(r.success_rate)
        m["latency"].append(r.avg_latency_ms)
        m["speedup"].append(r.speedup_vs_full)
        m["prefix"].append(r.avg_accepted_prefix)
        m["fallback"].append(r.fallback_rate)

    for method, vals in methods.items():
        data["aggregated"][method] = {
            "success_mean": float(np.mean(vals["success"])),
            "success_std": float(np.std(vals["success"])),
            "latency_mean": float(np.mean(vals["latency"])),
            "speedup_mean": float(np.mean(vals["speedup"])),
            "prefix_mean": float(np.mean(vals["prefix"])),
            "fallback_mean": float(np.mean(vals["fallback"])),
        }

    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Metrics JSON: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to benchmark_results.json",
    )
    parser.add_argument(
        "--output",
        default="./experiments/report",
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = load_results(args.results)
    print(f"Loaded {len(results)} results from {args.results}")

    generate_main_csv(results, args.output)
    generate_main_markdown(results, args.output)
    generate_latency_breakdown_csv(results, args.output)
    generate_metrics_json(results, args.output)

    print(f"\nAll reports saved to {args.output}")
    print("Files: main_table.csv, main_table.md, latency_breakdown.csv, metrics.json")


if __name__ == "__main__":
    main()
