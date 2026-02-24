"""
Experiment 1: Compression Ratio

Measures token reduction across 5 realistic agent scenarios.
Tests each layer individually and combined to show per-layer breakdown.
No API calls needed.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/home/keith/contextprune")

from contextprune.tokenizer import (
    count_message_tokens,
    count_system_tokens,
    count_tools_tokens,
)
from contextprune.dedup import SemanticDeduplicator
from contextprune.tool_filter import ToolSchemaFilter
from contextprune.budget import TokenBudgetInjector

from benchmarks.scenarios import get_all_scenarios


def count_all(
    messages: List[Dict[str, Any]],
    system: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
) -> int:
    return (
        count_message_tokens(messages)
        + count_system_tokens(system)
        + count_tools_tokens(tools)
    )


def run_pipeline(
    system: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    similarity_threshold: float = 0.85,
    max_tools: int = 10,
) -> Dict[str, Any]:
    """
    Run the full 3-layer compression pipeline and return per-layer stats.
    """
    dedup = SemanticDeduplicator(similarity_threshold=similarity_threshold)
    tool_filter = ToolSchemaFilter(max_tools=max_tools)
    budget_injector = TokenBudgetInjector()

    result: Dict[str, Any] = {}

    # Baseline
    orig_tokens = count_all(messages, system, tools)
    result["original_tokens"] = orig_tokens

    # --- Layer 1: Semantic Deduplication ---
    t0 = time.perf_counter()
    new_messages, new_system, dedup_removed = dedup.deduplicate(messages, system=system)
    t1 = time.perf_counter()

    after_dedup = count_all(new_messages, new_system, tools)
    result["after_dedup_tokens"] = after_dedup
    result["dedup_removed_sentences"] = dedup_removed
    result["dedup_reduction_pct"] = round(
        max(0.0, (orig_tokens - after_dedup) / orig_tokens * 100) if orig_tokens else 0, 1
    )
    result["dedup_ms"] = round((t1 - t0) * 1000, 2)

    # --- Layer 2: Tool Filtering ---
    t0 = time.perf_counter()
    filtered_tools = tools
    tools_removed = 0
    if tools and len(tools) > max_tools:
        filtered_tools, tools_removed = tool_filter.filter(tools, new_messages)
    t1 = time.perf_counter()

    after_tools = count_all(new_messages, new_system, filtered_tools)
    result["after_tools_tokens"] = after_tools
    result["tools_removed"] = tools_removed
    result["tools_reduction_pct"] = round(
        max(0.0, (after_dedup - after_tools) / orig_tokens * 100) if orig_tokens else 0, 1
    )
    result["tools_ms"] = round((t1 - t0) * 1000, 2)

    # --- Layer 3: Budget Injection ---
    t0 = time.perf_counter()
    new_system_budgeted, budget_injected = budget_injector.inject(new_system, new_messages)
    t1 = time.perf_counter()

    after_budget = count_all(new_messages, new_system_budgeted, filtered_tools)
    result["after_budget_tokens"] = after_budget
    result["budget_injected"] = budget_injected
    result["budget_ms"] = round((t1 - t0) * 1000, 2)

    # Final
    result["compressed_tokens"] = after_budget
    reduction = (orig_tokens - after_budget) / orig_tokens * 100 if orig_tokens else 0
    result["reduction_pct"] = round(max(0.0, reduction), 1)
    result["total_ms"] = round(result["dedup_ms"] + result["tools_ms"] + result["budget_ms"], 2)

    return result


def run_exp1() -> List[Dict[str, Any]]:
    """Run Experiment 1 across all scenarios."""
    scenarios = get_all_scenarios()
    results = []

    for name, system, messages, tools in scenarios:
        t_start = time.perf_counter()
        stats = run_pipeline(system, messages, tools)
        t_end = time.perf_counter()
        stats["scenario"] = name
        stats["wall_ms"] = round((t_end - t_start) * 1000, 2)
        results.append(stats)

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    """Print Exp 1 results as a formatted table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Compression Ratio")
    print("=" * 80)
    print(f"\n{'Scenario':<22} {'Before':>8} {'After':>8} {'Reduction':>10} {'Dedup%':>8} {'Tools%':>8} {'Time(ms)':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['scenario']:<22} {r['original_tokens']:>8,} {r['compressed_tokens']:>8,} "
            f"{r['reduction_pct']:>9.1f}% {r['dedup_reduction_pct']:>7.1f}% "
            f"{r['tools_reduction_pct']:>7.1f}% {r['total_ms']:>10.1f}"
        )
    print()
    print("Per-layer breakdown:")
    print(f"{'Scenario':<22} {'Orig':>8} {'→Dedup':>8} {'→Tools':>8} {'→Budget':>8} {'Sentences':>10} {'ToolsRm':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['scenario']:<22} {r['original_tokens']:>8,} "
            f"{r['after_dedup_tokens']:>8,} {r['after_tools_tokens']:>8,} "
            f"{r['after_budget_tokens']:>8,} {r['dedup_removed_sentences']:>10} "
            f"{r['tools_removed']:>8}"
        )


if __name__ == "__main__":
    results = run_exp1()
    print_results(results)
