"""
Baseline Comparison Harness
============================

Runs the same compression task through 4 conditions and prints a side-by-side table:
  1. Raw          — no compression
  2. Truncation   — cut tokens from end to match LLMLingua-2 output length
  3. LLMLingua-2  — Microsoft's baseline (Pan et al., ACL Findings 2024)
  4. ContextPrune — our system (semantic dedup + tool filter + budget injection)

Usage:
    cd /home/keith/contextprune
    .venv/bin/python3 benchmarks/compare_baselines.py

Optional flags:
    --rate 0.5          LLMLingua-2 compression rate (default: 0.5)
    --runs 5            Number of latency timing runs per scenario (default: 5)
    --skip-llmlingua    Skip LLMLingua-2 (faster, no model download needed)
    --skip-contextprune Skip ContextPrune (no embedding model needed)
    --verbose           Enable detailed logging
"""

from __future__ import annotations

import argparse
import logging
import statistics
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def truncate_to_budget(
    messages: List[Dict[str, Any]],
    system: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    target_tokens: int,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[List[Dict[str, Any]]]]:
    """Truncate messages from the end until token count <= target_tokens.

    Removes entire messages from the end of the list. System and tools are
    kept intact (truncation is the simplest possible baseline — just cut content).
    """
    if count_all(messages, system, tools) <= target_tokens:
        return messages, system, tools

    # Try removing messages from the end one at a time
    truncated = list(messages)
    while truncated and count_all(truncated, system, tools) > target_tokens:
        truncated.pop()

    return truncated, system, tools


def run_contextprune(
    system: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Run ContextPrune's 3-layer pipeline and return compressed output + stats."""
    dedup = SemanticDeduplicator(similarity_threshold=0.85)
    tool_filter = ToolSchemaFilter(max_tools=10)
    budget_injector = TokenBudgetInjector()

    orig_tokens = count_all(messages, system, tools)

    # Layer 1: Semantic dedup
    new_messages, new_system, _ = dedup.deduplicate(messages, system=system)

    # Layer 2: Tool filtering
    filtered_tools = tools
    if tools and len(tools) > 10:
        filtered_tools, _ = tool_filter.filter(tools, new_messages)

    # Layer 3: Budget injection
    new_system_budgeted, _ = budget_injector.inject(new_system, new_messages)

    comp_tokens = count_all(new_messages, new_system_budgeted, filtered_tools)
    reduction = (orig_tokens - comp_tokens) / orig_tokens * 100 if orig_tokens else 0

    stats = {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "reduction_pct": round(max(0.0, reduction), 1),
        "method": "contextprune",
    }
    return new_messages, new_system_budgeted, filtered_tools, stats


# ---------------------------------------------------------------------------
# Latency measurement helpers
# ---------------------------------------------------------------------------

def median_ms(fn, n: int = 5) -> float:
    """Run fn() n times and return the median wall-clock time in milliseconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def fmt_cell(tokens: int, reduction_pct: Optional[float] = None) -> str:
    """Format a table cell. reduction_pct is the % tokens REMOVED (positive = smaller)."""
    if reduction_pct is None:
        return f"{tokens:,}t"
    return f"{tokens:,}t (-{reduction_pct:.0f}%)"


def print_comparison_table(rows: List[Dict[str, Any]]) -> None:
    col_w = [22, 10, 14, 14, 14]
    header = ["Scenario", "Raw", "Truncation", "LLMLingua-2", "ContextPrune"]

    sep = "-" * (sum(col_w) + len(col_w) * 3 + 1)
    header_row = (
        f" {header[0]:<{col_w[0]}} | {header[1]:<{col_w[1]}} | {header[2]:<{col_w[2]}} | "
        f"{header[3]:<{col_w[3]}} | {header[4]:<{col_w[4]}}"
    )

    print()
    print("Compression Comparison")
    print("======================")
    print(header_row)
    print(sep)

    print("  (percentages = token reduction vs raw baseline)")
    print()
    for r in rows:
        raw_cell = fmt_cell(r["raw_tokens"])

        trunc_pct = r.get("trunc_pct")  # reduction_pct
        trunc_cell = fmt_cell(r["trunc_tokens"], trunc_pct)

        ll2_tok = r.get("ll2_tokens")
        ll2_pct = r.get("ll2_pct")  # reduction_pct
        if ll2_tok is None:
            ll2_cell = "FAILED"
        else:
            ll2_cell = fmt_cell(ll2_tok, ll2_pct)

        cp_tok = r.get("cp_tokens")
        cp_pct = r.get("cp_pct")  # reduction_pct
        if cp_tok is None:
            cp_cell = "SKIPPED"
        else:
            cp_cell = fmt_cell(cp_tok, cp_pct)

        print(
            f" {r['scenario']:<{col_w[0]}} | {raw_cell:<{col_w[1]}} | {trunc_cell:<{col_w[2]}} | "
            f"{ll2_cell:<{col_w[3]}} | {cp_cell:<{col_w[4]}}"
        )

    print()


def print_latency_table(latencies: Dict[str, Optional[float]], n_runs: int) -> None:
    print(f"Latency (median {n_runs} runs)")
    print("=" * 32)
    for method, ms in latencies.items():
        if ms is None:
            print(f"  {method:<15} FAILED/SKIPPED")
        else:
            print(f"  {method:<15} {ms:.2f}ms")
    print()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_comparison(
    rate: float = 0.5,
    n_runs: int = 5,
    skip_llmlingua: bool = False,
    skip_contextprune: bool = False,
    verbose: bool = False,
) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO)

    scenarios = get_all_scenarios()  # [(name, system, messages, tools), ...]

    # Pre-load LLMLingua-2 model once (amortise the 10-30s download across all scenarios)
    ll2_baseline = None
    ll2_load_error: Optional[str] = None

    if not skip_llmlingua:
        try:
            from contextprune.baselines import LLMLingua2Baseline
            print(f"Loading LLMLingua-2 model (rate={rate}) — may take 10-30s on first run...")
            ll2_baseline = LLMLingua2Baseline(rate=rate)
            # Trigger model load now so latency timings are fair
            ll2_baseline._load_compressor()
            print("LLMLingua-2 model ready.\n")
        except ImportError as e:
            ll2_load_error = str(e)
            print(f"⚠ LLMLingua-2 not available: {e}\n")
        except Exception as e:
            ll2_load_error = str(e)
            print(f"⚠ LLMLingua-2 model failed to load: {e}\n")

    table_rows = []
    latency_trunc_all = []
    latency_ll2_all = []
    latency_cp_all = []

    for name, system, messages, tools in scenarios:
        print(f"Running scenario: {name} ...")
        raw_tokens = count_all(messages, system, tools)

        # ----- LLMLingua-2 -----
        ll2_tokens: Optional[int] = None
        ll2_pct: Optional[float] = None

        if ll2_baseline is not None:
            try:
                comp_msgs, comp_sys, ll2_stats = ll2_baseline.compress(messages, system)
                # Use our tokenizer for consistent measurement across all methods.
                # LLMLingua-2 uses BERT tokenization internally; we normalize everything
                # through tiktoken (cl100k) so numbers are comparable on the same scale.
                # Tools are NOT compressed by LLMLingua-2 — add them back unchanged.
                ll2_msg_tokens = count_message_tokens(comp_msgs)
                ll2_sys_tokens = count_system_tokens(comp_sys)
                ll2_tool_tokens = count_tools_tokens(tools)   # pass-through unchanged
                ll2_tokens = ll2_msg_tokens + ll2_sys_tokens + ll2_tool_tokens
                ll2_pct = round(
                    (raw_tokens - ll2_tokens) / raw_tokens * 100, 1
                ) if raw_tokens else 0.0
            except Exception as e:
                print(f"  LLMLingua-2 failed on {name}: {e}")

        # ----- Truncation (target = 50% of raw, same as the LLMLingua-2 target rate) -----
        # We target 50% to give truncation the same budget as LLMLingua-2's rate.
        target_trunc = raw_tokens // 2
        trunc_msgs, trunc_sys, trunc_tools = truncate_to_budget(
            messages, system, tools, target_trunc
        )
        trunc_tokens = count_all(trunc_msgs, trunc_sys, trunc_tools)
        trunc_pct = round(
            (raw_tokens - trunc_tokens) / raw_tokens * 100, 1
        ) if raw_tokens else 0.0

        # ----- ContextPrune -----
        cp_tokens: Optional[int] = None
        cp_pct: Optional[float] = None

        if not skip_contextprune:
            try:
                _, _, _, cp_stats = run_contextprune(system, messages, tools)
                cp_tokens = cp_stats["compressed_tokens"]
                cp_pct = cp_stats["reduction_pct"]
            except Exception as e:
                print(f"  ContextPrune failed on {name}: {e}")

        table_rows.append({
            "scenario": name,
            "raw_tokens": raw_tokens,
            "trunc_tokens": trunc_tokens,
            "trunc_pct": trunc_pct,
            "ll2_tokens": ll2_tokens,
            "ll2_pct": ll2_pct,
            "cp_tokens": cp_tokens,
            "cp_pct": cp_pct,
        })

    # ----- Latency measurement (first scenario only — representative) -----
    print(f"\nMeasuring latency on '{scenarios[0][0]}' ({n_runs} runs each)...")
    name0, sys0, msgs0, tools0 = scenarios[0]
    raw0 = count_all(msgs0, sys0, tools0)
    target0 = raw0 // 2

    # Truncation latency
    def _trunc():
        truncate_to_budget(msgs0, sys0, tools0, target0)

    trunc_ms = median_ms(_trunc, n_runs)
    latency_trunc_all.append(trunc_ms)

    # LLMLingua-2 latency
    ll2_ms: Optional[float] = None
    if ll2_baseline is not None:
        try:
            def _ll2():
                ll2_baseline.compress(msgs0, sys0)
            ll2_ms = median_ms(_ll2, n_runs)
        except Exception as e:
            print(f"  LLMLingua-2 latency measurement failed: {e}")

    # ContextPrune latency
    cp_ms: Optional[float] = None
    if not skip_contextprune:
        try:
            def _cp():
                run_contextprune(sys0, msgs0, tools0)
            cp_ms = median_ms(_cp, n_runs)
        except Exception as e:
            print(f"  ContextPrune latency measurement failed: {e}")

    # ----- Output -----
    print_comparison_table(table_rows)

    latencies = {
        "Truncation": trunc_ms,
        "LLMLingua-2": ll2_ms,
        "ContextPrune": cp_ms,
    }
    print_latency_table(latencies, n_runs)

    # Notes
    if ll2_load_error:
        print(f"Note: LLMLingua-2 failed to load — {ll2_load_error}")
        print("      Install with: pip install llmlingua")
    if skip_llmlingua:
        print("Note: LLMLingua-2 was skipped (--skip-llmlingua)")
    if skip_contextprune:
        print("Note: ContextPrune was skipped (--skip-contextprune)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare compression baselines")
    parser.add_argument("--rate", type=float, default=0.5, help="LLMLingua-2 compression rate (default: 0.5)")
    parser.add_argument("--runs", type=int, default=5, help="Number of latency timing runs (default: 5)")
    parser.add_argument("--skip-llmlingua", action="store_true", help="Skip LLMLingua-2")
    parser.add_argument("--skip-contextprune", action="store_true", help="Skip ContextPrune")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    run_comparison(
        rate=args.rate,
        n_runs=args.runs,
        skip_llmlingua=args.skip_llmlingua,
        skip_contextprune=args.skip_contextprune,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
