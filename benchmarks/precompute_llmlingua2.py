"""
Pre-compute LLMLingua-2 compressed inputs for all datasets.
Run this ONCE overnight before starting Exp 3.

Saves compressed inputs to benchmarks/data/<dataset>/llmlingua2_compressed.jsonl

Why: LLMLingua-2 takes ~30s/question on CPU. Pre-computing means the live
API experiment runs at full speed with no local inference overhead.

Usage:
  # Dry run (show what would be compressed, no actual compression)
  python3 benchmarks/precompute_llmlingua2.py --dry-run

  # Run all datasets
  python3 benchmarks/precompute_llmlingua2.py --datasets all

  # Run one dataset
  python3 benchmarks/precompute_llmlingua2.py --datasets mmlu_pro

  # Run multiple datasets
  python3 benchmarks/precompute_llmlingua2.py --datasets mmlu_pro,math500

Progress bar with tqdm. Checkpoints after every 10 items (resumes if killed).
Output: one JSONL per dataset.
Each line = {id, original_tokens, compressed_tokens, compressed_messages, compressed_system}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent / "data"

VALID_DATASETS = ["mmlu_pro", "math500", "livecodebench", "frames", "gaia"]

COMPRESSION_RATIO = 0.45  # match exp3 default


# ---------------------------------------------------------------------------
# Cache path helper
# ---------------------------------------------------------------------------

def cache_path(dataset: str) -> Path:
    return DATA_DIR / dataset / "llmlingua2_compressed.jsonl"


# ---------------------------------------------------------------------------
# Load existing cache (for resume)
# ---------------------------------------------------------------------------

def load_cached_ids(dataset: str) -> set[str]:
    """Return set of item IDs already in the cache."""
    path = cache_path(dataset)
    if not path.exists():
        return set()
    cached: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cached.add(record["id"])
            except Exception:
                pass
    return cached


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _messages_to_text(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role.upper()}]: {content}")
    return "\n\n".join(parts)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _count_message_tokens(messages: list[dict]) -> int:
    return sum(_approx_tokens(m.get("content", "")) for m in messages)


# ---------------------------------------------------------------------------
# Compression logic
# ---------------------------------------------------------------------------

def compress_item(
    item_id: str,
    messages: list[dict],
    system: str,
    compressor: Any,
    compression_ratio: float = COMPRESSION_RATIO,
) -> dict:
    """
    Compress one item with LLMLingua-2.
    Returns a cache record ready to write to JSONL.
    """
    original_tokens = _count_message_tokens(messages) + _approx_tokens(system)

    context_msgs = messages[:-1]
    question_msg = messages[-1] if messages else {"role": "user", "content": ""}
    context_text = _messages_to_text(context_msgs)
    question_text = question_msg.get("content", "")

    compressed_messages = messages
    compressed_system = system

    if len(context_text) > 50:
        result = compressor.compress_prompt(
            context_text,
            instruction=question_text,
            question=question_text,
            rate=compression_ratio,
        )
        compressed_context = result["compressed_prompt"]
        compressed_messages = [
            {"role": "system", "content": compressed_context},
            question_msg,
        ]
    else:
        compressed_messages = messages

    compressed_tokens = _count_message_tokens(compressed_messages) + _approx_tokens(compressed_system)

    return {
        "id": item_id,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": round(compressed_tokens / max(1, original_tokens), 4),
        "compressed_messages": compressed_messages,
        "compressed_system": compressed_system,
    }


# ---------------------------------------------------------------------------
# Dataset loader (minimal — reuses injector)
# ---------------------------------------------------------------------------

def load_and_inject(dataset: str, limit: int | None = None) -> list[dict]:
    """Load dataset items and inject agent context."""
    from benchmarks.context_injector import ContextInjector

    path = DATA_DIR / dataset / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run: python3 benchmarks/data/download_datasets.py --dataset {dataset}"
        )

    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    if limit:
        items = items[:limit]

    injector = ContextInjector()
    return [injector.inject(item, context_size="medium") for item in items]


# ---------------------------------------------------------------------------
# Main precompute logic
# ---------------------------------------------------------------------------

def precompute_dataset(
    dataset: str,
    dry_run: bool = False,
    checkpoint_every: int = 10,
) -> None:
    """Pre-compute LLMLingua-2 compression for one dataset."""
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback progress if tqdm not installed
        def tqdm(iterable, **kwargs):  # type: ignore
            total = kwargs.get("total", "?")
            desc = kwargs.get("desc", "")
            for i, item in enumerate(iterable):
                print(f"\r  {desc}: {i+1}/{total}", end="", flush=True)
                yield item
            print()

    print(f"\n[{dataset}]")

    # Check dataset exists
    data_path = DATA_DIR / dataset / "test.jsonl"
    if not data_path.exists():
        print(f"  SKIP — dataset not found at {data_path}")
        print(f"  Run: python3 benchmarks/data/download_datasets.py --dataset {dataset}")
        return

    # Count items
    with open(data_path, encoding="utf-8") as f:
        total_items = sum(1 for line in f if line.strip())

    print(f"  Found {total_items:,} items in {data_path}")

    if dry_run:
        print(f"  DRY RUN — would compress {total_items:,} items")
        print(f"  Output: {cache_path(dataset)}")
        print(f"  Estimated time: ~{total_items * 30 / 60:.0f} minutes on CPU (30s/item)")
        print(f"  Estimated time: ~{total_items * 3 / 60:.0f} minutes on GPU (3s/item)")
        return

    # Load LLMLingua-2
    print("  Loading LLMLingua-2 model (first run downloads ~1.5GB)…")
    try:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        print("  LLMLingua-2 loaded.")
    except ImportError:
        print("  ERROR: llmlingua not installed.")
        print("  Run: .venv/bin/pip install llmlingua")
        return
    except Exception as e:
        print(f"  ERROR loading LLMLingua-2: {e}")
        return

    # Load cached IDs (for resume)
    cached_ids = load_cached_ids(dataset)
    if cached_ids:
        print(f"  Resuming — {len(cached_ids):,} items already cached.")

    # Load and inject all items
    print("  Loading and injecting context…")
    items = load_and_inject(dataset)
    items_to_process = [i for i in items if i.get("id", "") not in cached_ids]
    print(f"  {len(items_to_process):,} items to process.")

    if not items_to_process:
        print("  All items already cached. Nothing to do.")
        return

    # Open cache file (append mode)
    out_path = cache_path(dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buffer: list[dict] = []
    errors = 0
    t_start = time.time()

    with open(out_path, "a", encoding="utf-8") as f_out:
        for i, item in enumerate(tqdm(items_to_process, desc=f"  {dataset}", total=len(items_to_process))):
            item_id = item.get("id", f"item_{i:05d}")
            messages = item.get("messages", [])
            system = item.get("system", "")

            try:
                record = compress_item(item_id, messages, system, compressor)
                buffer.append(record)
            except Exception as e:
                errors += 1
                print(f"\n  WARN: failed on {item_id}: {e}")
                # Write a placeholder so we don't retry a broken item
                buffer.append({
                    "id": item_id,
                    "error": str(e),
                    "original_tokens": 0,
                    "compressed_tokens": 0,
                    "compression_ratio": 1.0,
                    "compressed_messages": messages,
                    "compressed_system": system,
                })

            # Checkpoint every N items
            if len(buffer) >= checkpoint_every:
                for rec in buffer:
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                buffer.clear()

        # Flush remaining
        if buffer:
            for rec in buffer:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start
    processed = len(items_to_process)
    rate = processed / max(1, elapsed)
    print(f"\n  Done: {processed:,} items in {elapsed:.1f}s ({rate:.1f} items/s)")
    if errors:
        print(f"  Warnings: {errors} items failed (logged with error field)")
    print(f"  Cache written → {out_path}")

    # Summary stats
    all_cached = load_cached_ids(dataset)
    print(f"  Total cached: {len(all_cached):,} / {total_items:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute LLMLingua-2 compressed inputs for Experiment 3."
    )
    p.add_argument(
        "--datasets", default="all",
        help="Comma-separated dataset names or 'all'. Valid: " + ", ".join(VALID_DATASETS),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be compressed without doing any actual compression.",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Write checkpoint after this many items (default: 10).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.datasets.strip().lower() == "all":
        datasets = VALID_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        for d in datasets:
            if d not in VALID_DATASETS:
                print(f"Unknown dataset: {d!r}. Valid: {', '.join(VALID_DATASETS)}")
                sys.exit(1)

    print(f"=== LLMLingua-2 Pre-computation ===")
    print(f"  Datasets: {datasets}")
    if args.dry_run:
        print("  Mode: DRY RUN (no actual compression)")
    else:
        print("  Mode: LIVE (will download model + compress)")
    print(f"  Checkpoint every: {args.checkpoint_every} items")

    for dataset in datasets:
        precompute_dataset(
            dataset,
            dry_run=args.dry_run,
            checkpoint_every=args.checkpoint_every,
        )

    print("\n=== Pre-computation complete ===")
    if not args.dry_run:
        print("  You can now run exp3_accuracy.py — LLMLingua-2 will use the cache.")


if __name__ == "__main__":
    main()
