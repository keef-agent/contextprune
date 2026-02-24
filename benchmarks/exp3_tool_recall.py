"""
Experiment 3: Tool Recall & Precision

10 test cases: user query + correct tool(s) + pool of 20 tools (2-3 relevant, rest noise).
Runs ToolSchemaFilter with max_tools=5.

Metrics:
  Recall:    fraction of correct tools that appear in the filtered set (target: 100%)
  Precision: fraction of the filtered set that are actually correct for the query
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Set

sys.path.insert(0, "/home/keith/contextprune")

from contextprune.tool_filter import ToolSchemaFilter
from benchmarks.scenarios import TOOL_POOL_RECALL, RECALL_TEST_CASES


def run_exp3(max_tools: int = 5) -> List[Dict[str, Any]]:
    """Run Experiment 3 — tool filter recall and precision."""
    tool_filter = ToolSchemaFilter(max_tools=max_tools)
    results = []

    for case in RECALL_TEST_CASES:
        query = case["query"]
        correct_tools: List[str] = case["correct_tools"]
        messages: List[Dict[str, Any]] = case["messages"]

        # Filter the 20-tool pool
        filtered, removed = tool_filter.filter(TOOL_POOL_RECALL, messages)
        filtered_names: Set[str] = {t["name"] for t in filtered}
        correct_set: Set[str] = set(correct_tools)

        # Recall: did we keep all the correct tools?
        recall_hits = correct_set & filtered_names
        recall = len(recall_hits) / len(correct_set) if correct_set else 0.0

        # Precision: of the tools we kept, how many are actually relevant?
        precision = len(recall_hits) / len(filtered_names) if filtered_names else 0.0

        results.append({
            "query": query[:60] + "..." if len(query) > 60 else query,
            "correct_tools": correct_tools,
            "filtered_tools": sorted(filtered_names),
            "filtered_count": len(filtered_names),
            "removed_count": removed,
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "correct_found": sorted(recall_hits),
            "missed": sorted(correct_set - filtered_names),
        })

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Tool Recall & Precision")
    print("=" * 80)
    total_recall = sum(r["recall"] for r in results) / len(results)
    total_precision = sum(r["precision"] for r in results) / len(results)

    print(f"\n{'Query':<45} {'Correct':>12} {'Recall':>8} {'Precision':>10} {'Missed'}")
    print("-" * 100)
    for r in results:
        missed_str = ", ".join(r["missed"]) if r["missed"] else "none"
        correct_str = ", ".join(r["correct_tools"])
        print(
            f"{r['query']:<45} {correct_str:>12} {r['recall']:>8.1%} {r['precision']:>10.1%}  {missed_str}"
        )
    print("-" * 100)
    print(f"{'AVERAGE':<45} {'':>12} {total_recall:>8.1%} {total_precision:>10.1%}")
    print()
    print(f"Mean Recall:    {total_recall:.1%}")
    print(f"Mean Precision: {total_precision:.1%}")


if __name__ == "__main__":
    results = run_exp3()
    print_results(results)
