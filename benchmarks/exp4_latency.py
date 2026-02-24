"""
Experiment 4: Latency Overhead

For each Exp 1 scenario, run compression 10x, take median.
Simulate API call with time.sleep(0.5).
Target: overhead < 5%.
"""

from __future__ import annotations

import sys
import time
from statistics import median
from typing import Any, Dict, List

sys.path.insert(0, "/home/keith/contextprune")

from contextprune.dedup import SemanticDeduplicator
from contextprune.tool_filter import ToolSchemaFilter
from contextprune.budget import TokenBudgetInjector
from benchmarks.scenarios import get_all_scenarios

SIMULATED_API_MS = 500.0  # 0.5 second simulated network + inference latency
RUNS_PER_SCENARIO = 10


def compress_once(
    system: str,
    messages: List[Dict[str, Any]],
    tools: Any,
    similarity_threshold: float = 0.85,
    max_tools: int = 10,
) -> float:
    """Run the full compression pipeline once, return elapsed ms."""
    dedup = SemanticDeduplicator(similarity_threshold=similarity_threshold)
    tool_filter = ToolSchemaFilter(max_tools=max_tools)
    budget_injector = TokenBudgetInjector()

    t0 = time.perf_counter()

    new_messages, new_system, _ = dedup.deduplicate(messages, system=system)

    if tools and len(tools) > max_tools:
        filtered_tools, _ = tool_filter.filter(tools, new_messages)
    else:
        filtered_tools = tools

    budget_injector.inject(new_system, new_messages)

    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def run_exp4() -> List[Dict[str, Any]]:
    """Run Experiment 4 — latency benchmarking."""
    scenarios = get_all_scenarios()
    results = []

    for name, system, messages, tools in scenarios:
        # Warm up
        compress_once(system, messages, tools)

        # Run N times, collect timings
        timings = []
        for _ in range(RUNS_PER_SCENARIO):
            ms = compress_once(system, messages, tools)
            timings.append(ms)

        med_ms = median(timings)
        min_ms = min(timings)
        max_ms = max(timings)

        # Simulated total latency
        total_ms = med_ms + SIMULATED_API_MS
        overhead_pct = (med_ms / total_ms) * 100.0

        results.append({
            "scenario": name,
            "compression_median_ms": round(med_ms, 2),
            "compression_min_ms": round(min_ms, 2),
            "compression_max_ms": round(max_ms, 2),
            "simulated_api_ms": SIMULATED_API_MS,
            "total_ms": round(total_ms, 2),
            "overhead_pct": round(overhead_pct, 2),
            "meets_target": overhead_pct < 5.0,
            "runs": RUNS_PER_SCENARIO,
        })

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Latency Overhead")
    print(f"(Simulated API latency: {SIMULATED_API_MS}ms, {RUNS_PER_SCENARIO} runs/scenario)")
    print("=" * 80)
    print(f"\n{'Scenario':<22} {'Compress(ms)':>13} {'API(ms)':>8} {'Total(ms)':>10} {'Overhead%':>10} {'<5%?':>6}")
    print("-" * 75)
    for r in results:
        target = "✓" if r["meets_target"] else "✗"
        print(
            f"{r['scenario']:<22} {r['compression_median_ms']:>13.2f} "
            f"{r['simulated_api_ms']:>8.0f} {r['total_ms']:>10.2f} "
            f"{r['overhead_pct']:>9.2f}% {target:>6}"
        )
    avg_overhead = sum(r["overhead_pct"] for r in results) / len(results)
    all_pass = all(r["meets_target"] for r in results)
    print("-" * 75)
    print(f"{'AVERAGE':<22} {'':>13} {'':>8} {'':>10} {avg_overhead:>9.2f}%")
    print(f"\nAll scenarios meet <5% overhead target: {'YES ✓' if all_pass else 'NO ✗'}")


if __name__ == "__main__":
    results = run_exp4()
    print_results(results)
