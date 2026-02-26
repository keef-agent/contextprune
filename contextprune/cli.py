"""CLI entry point for contextprune."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .budget import TokenBudgetInjector
from .dedup import SemanticDeduplicator
from .stats import CompressionStats
from .tokenizer import (
    count_message_tokens,
    count_system_tokens,
    count_tools_tokens,
)
from .tool_filter import ToolSchemaFilter


def _profile(input_path: str, verbose: bool = False) -> None:
    """Profile a conversation JSON file and show token savings."""
    with open(input_path) as f:
        data = json.load(f)

    messages: List[Dict[str, Any]] = data.get("messages", [])
    system: Optional[str] = data.get("system", None)
    tools: Optional[List[Dict[str, Any]]] = data.get("tools", None)

    # Original counts
    msg_tokens = count_message_tokens(messages)
    sys_tokens = count_system_tokens(system)
    tool_tokens = count_tools_tokens(tools)
    original_total = msg_tokens + sys_tokens + tool_tokens

    print(f"Input: {input_path}")
    print(f"Messages: {len(messages)}")
    print(f"Tools: {len(tools) if tools else 0}")
    print()
    print("--- Original Token Breakdown ---")
    print(f"  Messages:     {msg_tokens:>8}")
    print(f"  System:       {sys_tokens:>8}")
    print(f"  Tools:        {tool_tokens:>8}")
    print(f"  Total:        {original_total:>8}")
    print()

    # Run each layer
    savings_breakdown = []

    # 1. Dedup
    dedup = SemanticDeduplicator(similarity_threshold=0.85)
    new_messages, new_system, removed = dedup.deduplicate(messages, system=system)
    dedup_msg_tokens = count_message_tokens(new_messages)
    dedup_sys_tokens = count_system_tokens(new_system)
    dedup_saved = (msg_tokens + sys_tokens) - (dedup_msg_tokens + dedup_sys_tokens)
    savings_breakdown.append(("SemanticDedup", dedup_saved, removed, "sentences removed"))

    # 2. Tool filter
    tool_saved = 0
    tools_removed = 0
    if tools and len(tools) > 10:
        tf = ToolSchemaFilter(max_tools=10)
        filtered_tools, tools_removed = tf.filter(tools, new_messages)
        new_tool_tokens = count_tools_tokens(filtered_tools)
        tool_saved = tool_tokens - new_tool_tokens
    savings_breakdown.append(("ToolFilter", tool_saved, tools_removed, "tools removed"))

    # 3. Budget injection (adds tokens, but saves on response side)
    budget = TokenBudgetInjector()
    _, injected = budget.inject(new_system, new_messages)
    savings_breakdown.append(("BudgetInjector", 0, 1 if injected else 0, "budget injected"))

    total_saved = dedup_saved + tool_saved
    compressed_total = original_total - total_saved

    print("--- Savings by Layer ---")
    for name, saved, count, unit in savings_breakdown:
        pct = (saved / original_total * 100) if original_total > 0 else 0
        print(f"  {name:<20} {saved:>6} tokens saved  ({count} {unit}, {pct:.1f}%)")
    print()
    print("--- Result ---")
    print(f"  Original:    {original_total:>8} tokens")
    print(f"  Compressed:  {compressed_total:>8} tokens")
    if original_total > 0:
        pct = (total_saved / original_total) * 100
        print(f"  Savings:     {pct:>7.1f}%")
    print()

    # Recommendations
    print("--- Recommendations ---")
    if removed == 0:
        print("  - Low redundancy detected. Consider if your context has repeated info.")
    if tools and len(tools) > 10:
        print(f"  - {tools_removed} tools filtered. Consider splitting tools across agents.")
    elif tools and len(tools) <= 10:
        print("  - Tool count is already low. No filtering needed.")
    if not tools:
        print("  - No tools in context. ToolFilter layer inactive.")
    if dedup_saved > original_total * 0.3:
        print("  - High redundancy! Consider restructuring your prompts to avoid repetition.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="contextprune",
        description="Analyze and reduce LLM API token usage",
    )
    subparsers = parser.add_subparsers(dest="command")

    profile_parser = subparsers.add_parser(
        "profile",
        help="Profile a conversation JSON file for token savings",
    )
    profile_parser.add_argument(
        "--input",
        required=True,
        help="Path to conversation JSON file",
    )
    profile_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the ContextPrune proxy server (drop-in Anthropic API proxy)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8899,
        help="Port to listen on (default: 8899)",
    )
    serve_parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Similarity threshold for deduplication (default: 0.82)",
    )
    serve_parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable stats file logging",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    if args.command == "profile":
        _profile(args.input, verbose=args.verbose)
    elif args.command == "serve":
        from .proxy import serve
        serve(
            port=args.port,
            threshold=args.threshold,
            enable_log=not args.no_log,
            host=args.host,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
