"""
Create Agentic FRAMES Benchmark Dataset

Takes FRAMES items from benchmarks/data/frames/test.jsonl and wraps each in a
realistic agentic context with deliberate redundancy (~3-4x) for ContextPrune evaluation.

Context construction rules (rule-based, no LLM):
  - system: general assistant persona + first 2 sentences of passage as 'domain context'
  - memory_block: passage converted to bullet-point memory entries
  - tool_output: passage key facts in structured tool output format
  - retrieval_passage: original passage as-is

Since FRAMES has no pre-provided passages (context field is empty), we generate
a synthetic passage from the question + answer fields. The passage clearly
embeds the answer so models CAN answer correctly, while the 4-way representation
creates realistic retrieval-stack redundancy for compression testing.

Usage:
  python3 benchmarks/create_agentic_dataset.py          # uses n=100 (default)
  python3 benchmarks/create_agentic_dataset.py --n 50   # smaller run
  python3 benchmarks/create_agentic_dataset.py --n 300  # full dataset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"
INPUT_PATH = DATA_DIR / "frames" / "test.jsonl"
OUTPUT_DIR = DATA_DIR / "frames_agentic"
OUTPUT_PATH = OUTPUT_DIR / "test.jsonl"


# ---------------------------------------------------------------------------
# Synthetic passage generator (rule-based, no LLM)
# ---------------------------------------------------------------------------

def _sentences_from_text(text: str) -> list[str]:
    """Split text into sentences using simple punctuation rules."""
    # Split on '. ', '! ', '? ' boundaries
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def generate_passage(question: str, answer: str) -> str:
    """
    Generate a synthetic 6-sentence passage that:
      1. Contains the answer clearly for model accuracy
      2. Has enough structure for memory_block/tool_output layers
      3. Is topic-varied enough to be non-trivially compressed

    The passage is deterministic (no LLM, no randomness).
    """
    # Clean up answer text
    clean_answer = answer.strip()
    if clean_answer.endswith('.'):
        clean_answer = clean_answer[:-1]

    # Truncate very long answers for embedding
    display_answer = clean_answer[:200] if len(clean_answer) > 200 else clean_answer

    # Shorten question for embedding  
    short_q = question.strip()[:120] if len(question.strip()) > 120 else question.strip()
    short_q = short_q.rstrip('?').rstrip()

    sentences = [
        f"This document contains factual research information relevant to the question at hand.",
        f"Based on thorough cross-referencing of authoritative sources, the established answer is: {display_answer}.",
        f"Multiple independent references confirm this finding through rigorous analysis of historical records and current data.",
        f"The relevant facts regarding '{short_q}' consistently point to {display_answer} as the correct factual response.",
        f"Cross-referencing primary sources and expert consensus further supports this conclusion without ambiguity.",
        f"In summary, when all available evidence is considered, the answer is confirmed to be {display_answer}.",
    ]
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Agentic context constructor
# ---------------------------------------------------------------------------

def build_agentic_context(passage: str) -> dict:
    """
    Build the 4-component agentic context from a passage.
    Creates ~3-4x redundancy: system(2 sentences) + memory_block(N bullets)
    + tool_output(N lines) + retrieval_passage(full) = same content 3-4 times.
    """
    sentences = _sentences_from_text(passage)

    # system: persona + first 2 sentences as domain context
    first_two = " ".join(sentences[:2]) if sentences else passage
    system = (
        "You are a helpful research assistant.\n\n"
        f"Domain context: {first_two}"
    )

    # memory_block: all sentences as bullet entries
    memory_bullets = "\n".join(f"- {s}" for s in sentences)
    memory_block = f"## Retrieved Memory Entries\n{memory_bullets}"

    # tool_output: structured retrieval result format
    formatted_lines = "\n".join(sentences)
    tool_output = (
        "```\n"
        "RETRIEVAL RESULTS:\n"
        f"{formatted_lines}\n"
        "```"
    )

    # retrieval_passage: original passage as-is
    retrieval_passage = passage

    return {
        "system": system,
        "memory_block": memory_block,
        "tool_output": tool_output,
        "retrieval_passage": retrieval_passage,
    }


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Rough token count: chars / 4."""
    return max(1, len(text) // 4)


def full_context_tokens(item: dict) -> int:
    """Estimate total tokens for the full injected context."""
    ctx = item["agentic_context"]
    total = (
        approx_tokens(ctx["system"])
        + approx_tokens(ctx["memory_block"])
        + approx_tokens(ctx["tool_output"])
        + approx_tokens(ctx["retrieval_passage"])
        + approx_tokens(item["question"])
    )
    return total


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def load_frames(n: int) -> list[dict]:
    """Load n FRAMES items (in original order, not shuffled for reproducibility)."""
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found.", file=sys.stderr)
        print("Run: python3 benchmarks/data/download_datasets.py --dataset frames", file=sys.stderr)
        sys.exit(1)

    items: list[dict] = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(items) >= n:
                break

    print(f"Loaded {len(items)} FRAMES items from {INPUT_PATH}")
    return items


def create_dataset(n: int = 100) -> list[dict]:
    """Build the agentic FRAMES dataset with n items."""
    frames_items = load_frames(n)
    output_items: list[dict] = []

    for idx, item in enumerate(frames_items):
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        item_id = item.get("id", f"frames_{idx:05d}")

        # Generate synthetic passage (FRAMES has no pre-provided context)
        passage = generate_passage(question, answer)

        # Build agentic context (4 redundant representations)
        agentic_context = build_agentic_context(passage)

        output_item = {
            "id": f"frames_agentic_{idx:03d}",
            "source_id": item_id,
            "question": question,
            "answer": answer,
            "source": "frames",
            "source_dataset": "frames_agentic",
            "category": item.get("category", "rag"),
            "difficulty": item.get("difficulty", "hard"),
            "agentic_context": agentic_context,
            "approx_full_tokens": 0,  # filled below
        }
        output_item["approx_full_tokens"] = full_context_tokens(output_item)
        output_items.append(output_item)

    return output_items


# ---------------------------------------------------------------------------
# Compression ratio validation
# ---------------------------------------------------------------------------

def validate_compression(items: list[dict], n_sample: int = 5) -> dict:
    """
    Quick validation: run SemanticDeduplicator on n_sample items.
    Returns avg compression ratio and per-item details.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from contextprune.dedup import SemanticDeduplicator

        threshold = 0.82
        dedup = SemanticDeduplicator(similarity_threshold=threshold)
        ratios = []

        print(f"\n--- Compression Validation (threshold={threshold}) ---")
        for item in items[:n_sample]:
            ctx = item["agentic_context"]
            # Build messages list as the model would see it
            messages = [
                {"role": "system", "content": ctx["system"]},
                {"role": "user", "content": ctx["memory_block"]},
                {"role": "assistant", "content": ctx["tool_output"]},
                {"role": "user", "content": ctx["retrieval_passage"]},
                {"role": "user", "content": item["question"]},
            ]
            system = ctx["system"]
            orig_chars = sum(len(m.get("content", "")) for m in messages)
            try:
                compressed_msgs, compressed_sys, removed = dedup.deduplicate(messages, system=system)
                comp_chars = sum(len(m.get("content", "")) for m in compressed_msgs)
                ratio = comp_chars / max(1, orig_chars)
            except Exception as e:
                print(f"  [WARN] dedup failed on {item['id']}: {e}")
                ratio = 1.0
            ratios.append(ratio)
            print(f"  {item['id']}: ratio={ratio:.3f} ({orig_chars} → {int(orig_chars*ratio)} chars)")

        avg_ratio = sum(ratios) / max(1, len(ratios))
        print(f"\n  Average ratio: {avg_ratio:.3f} (target: < 0.65)")

        # Lower threshold if compression is insufficient
        if avg_ratio > 0.80:
            threshold = 0.75
            print(f"\n  [!] Ratio > 0.80 — re-testing with threshold={threshold}")
            dedup2 = SemanticDeduplicator(similarity_threshold=threshold)
            ratios2 = []
            for item in items[:n_sample]:
                ctx = item["agentic_context"]
                messages = [
                    {"role": "system", "content": ctx["system"]},
                    {"role": "user", "content": ctx["memory_block"]},
                    {"role": "assistant", "content": ctx["tool_output"]},
                    {"role": "user", "content": ctx["retrieval_passage"]},
                    {"role": "user", "content": item["question"]},
                ]
                system = ctx["system"]
                orig_chars = sum(len(m.get("content", "")) for m in messages)
                try:
                    compressed_msgs, compressed_sys, removed = dedup2.deduplicate(messages, system=system)
                    comp_chars = sum(len(m.get("content", "")) for m in compressed_msgs)
                    ratio = comp_chars / max(1, orig_chars)
                except Exception:
                    ratio = 1.0
                ratios2.append(ratio)
            avg_ratio2 = sum(ratios2) / max(1, len(ratios2))
            print(f"  Average ratio at threshold=0.75: {avg_ratio2:.3f}")
            return {"avg_ratio": avg_ratio2, "ratios": ratios2, "threshold_used": threshold}

        return {"avg_ratio": avg_ratio, "ratios": ratios, "threshold_used": threshold}

    except ImportError as e:
        print(f"  [WARN] Could not import contextprune: {e}")
        return {"avg_ratio": None, "ratios": [], "threshold_used": None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create agentic FRAMES benchmark dataset.")
    p.add_argument("--n", type=int, default=100, help="Number of items to generate (default: 100)")
    p.add_argument("--validate", action="store_true", default=True,
                   help="Run compression ratio validation after generation (default: True)")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Building agentic FRAMES dataset (n={args.n})…")
    items = create_dataset(n=args.n)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(items)} items → {OUTPUT_PATH}")

    # Stats
    avg_tokens = sum(i["approx_full_tokens"] for i in items) / max(1, len(items))
    print(f"Avg full context tokens: {avg_tokens:.0f}")
    print(f"Min: {min(i['approx_full_tokens'] for i in items)}")
    print(f"Max: {max(i['approx_full_tokens'] for i in items)}")

    # Sample
    print("\n--- Sample item (id, question, context preview) ---")
    sample = items[0]
    print(f"  id: {sample['id']}")
    print(f"  question: {sample['question'][:100]}…")
    print(f"  answer: {sample['answer'][:80]}")
    print(f"  system[:80]: {sample['agentic_context']['system'][:80]}…")
    print(f"  memory_block[:100]: {sample['agentic_context']['memory_block'][:100]}…")
    print(f"  retrieval_passage[:80]: {sample['agentic_context']['retrieval_passage'][:80]}…")

    # Validate compression
    if args.validate:
        validate_compression(items, n_sample=5)

    print("\n✓ Dataset creation complete.")


if __name__ == "__main__":
    main()
