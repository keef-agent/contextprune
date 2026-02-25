#!/usr/bin/env python3
"""
CompressionSafety Manual Annotation Tool

Use your ChatGPT Pro (GPT-5.2) to annotate conversations for free.

WORKFLOW:
  1. Get the next batch prompt to paste into ChatGPT:
       python3 benchmarks/compression_safety_manual.py next-batch

  2. Paste the entire output into ChatGPT (GPT-5.2).
     Tell it: "Respond with ONLY the JSON array, no other text."

  3. Copy ChatGPT's response and save it to a file (e.g. /tmp/batch.json)
     Then ingest it:
       python3 benchmarks/compression_safety_manual.py ingest /tmp/batch.json

  4. Check progress:
       python3 benchmarks/compression_safety_manual.py status

Repeat until done. Each batch takes ~2 minutes in ChatGPT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SHAREGPT_PATH = REPO_ROOT / "benchmarks" / "data" / "sharegpt" / "test.jsonl"
OUT_DIR = REPO_ROOT / "benchmarks" / "data" / "compression_safety"
OUT_DIR.mkdir(exist_ok=True)

ANNOTATIONS_PATH = OUT_DIR / "annotations.jsonl"
PROGRESS_PATH = OUT_DIR / "manual_progress.json"
STATS_PATH = OUT_DIR / "stats.json"

MAX_SENTENCES = 25   # per conversation
BATCH_SIZE_DEFAULT = 5
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sharegpt(seed: int = SEED) -> list[dict]:
    items = []
    with open(SHAREGPT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    random.seed(seed)
    random.shuffle(items)
    return items


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"completed_ids": [], "batch_cursor": 0}


def _save_progress(progress: dict) -> None:
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def _load_annotations() -> list[dict]:
    if not ANNOTATIONS_PATH.exists():
        return []
    with open(ANNOTATIONS_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def _conversation_to_text(conversations: list) -> str:
    parts = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from", turn.get("role", "unknown"))
        text = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            parts.append(f"User: {text.strip()}")
        elif role in ("gpt", "assistant"):
            parts.append(f"Assistant: {text.strip()}")
    return "\n\n".join(parts)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping them meaningful."""
    # Split on period/!/? followed by space+capital or newline
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 20:  # skip fragments
            sentences.append(s)
    return sentences[:MAX_SENTENCES]


ANNOTATION_PROMPT_TEMPLATE = """You are a compression research annotator.

For each conversation below, annotate every sentence as:
- "essential": removing it would change the correct answer to the final question
- "redundant": removing it would NOT change the correct answer  
- "uncertain": unclear

Return ONLY a JSON array. No explanation, no markdown, no preamble.
One object per conversation. Format:

[
  {{
    "conversation_id": "conv_000",
    "annotations": [
      {{"sentence_id": 0, "label": "essential|redundant|uncertain", "reason": "one phrase"}},
      ...
    ]
  }},
  ...
]

---
CONVERSATIONS TO ANNOTATE:

{conversations_block}

---

Return ONLY the JSON array. No other text."""


def _format_conversation_block(items: list[dict]) -> str:
    blocks = []
    for item in items:
        conv_id = item["id"]
        conv_text = item["conv_text"]
        sentences = item["sentences"]
        final_q = item["final_q"]

        sentences_formatted = "\n".join(
            f"  [{i}] {s}" for i, s in enumerate(sentences)
        )

        block = f"""=== CONVERSATION {conv_id} ===
Full conversation:
{conv_text[:2000]}

Final question being answered:
{final_q[:300] or "(see last user turn above)"}

Sentences to label:
{sentences_formatted}"""
        blocks.append(block)

    return "\n\n" + ("-" * 60 + "\n\n").join(blocks)


def _prepare_items(conversations: list[dict], skip_ids: set) -> list[dict]:
    """Extract and structure conversations not yet annotated."""
    items = []
    for conv in conversations:
        raw_id = str(conv.get("id", ""))
        if raw_id in skip_ids:
            continue
        turns = conv.get("conversations", [])
        conv_text = _conversation_to_text(turns)
        sentences = _split_sentences(conv_text)
        if not sentences:
            continue

        # Final question = last human turn
        human_turns = [
            t for t in turns
            if isinstance(t, dict) and t.get("from", t.get("role", "")) in ("human", "user")
        ]
        final_q = human_turns[-1].get("value", human_turns[-1].get("content", "")) if human_turns else ""

        items.append({
            "id": raw_id,
            "conv_text": conv_text,
            "sentences": sentences,
            "final_q": final_q,
        })
    return items


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_next_batch(args) -> None:
    """Print the next batch prompt, ready to paste into ChatGPT."""
    conversations = _load_sharegpt()
    progress = _load_progress()
    completed_ids = set(progress["completed_ids"])

    remaining = _prepare_items(conversations, completed_ids)
    if not remaining:
        print("✅ All conversations annotated!")
        return

    batch_size = args.size
    batch = remaining[:batch_size]

    prompt = ANNOTATION_PROMPT_TEMPLATE.format(
        conversations_block=_format_conversation_block(batch)
    )

    # Save which IDs are in this batch so ingest can validate
    pending_path = OUT_DIR / "pending_batch.json"
    with open(pending_path, "w") as f:
        json.dump({
            "batch_ids": [item["id"] for item in batch],
            "batch_sentences": {item["id"]: item["sentences"] for item in batch},
        }, f)

    print("=" * 70)
    print(f"BATCH OF {len(batch)} CONVERSATIONS — paste everything below into ChatGPT")
    print(f"Remaining after this batch: {len(remaining) - len(batch)}")
    print("=" * 70)
    print()
    print(prompt)
    print()
    print("=" * 70)
    print(f"After ChatGPT responds:")
    print(f"  1. Copy the entire JSON array")
    print(f"  2. Save it to /tmp/batch_response.json")
    print(f"  3. Run: python3 benchmarks/compression_safety_manual.py ingest /tmp/batch_response.json")
    print("=" * 70)


def cmd_ingest(args) -> None:
    """Parse and save ChatGPT's JSON response."""
    response_path = Path(args.file)
    if not response_path.exists():
        print(f"❌ File not found: {response_path}")
        sys.exit(1)

    raw = response_path.read_text().strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print("First 300 chars of response:")
        print(raw[:300])
        sys.exit(1)

    if not isinstance(parsed, list):
        print(f"❌ Expected JSON array, got {type(parsed).__name__}")
        sys.exit(1)

    # Load pending batch info
    pending_path = OUT_DIR / "pending_batch.json"
    if pending_path.exists():
        with open(pending_path) as f:
            pending = json.load(f)
        expected_ids = set(pending["batch_ids"])
        batch_sentences = pending["batch_sentences"]
    else:
        expected_ids = None
        batch_sentences = {}

    progress = _load_progress()
    completed_ids = set(progress["completed_ids"])

    saved = 0
    skipped = 0
    errors = []

    for item in parsed:
        conv_id = str(item.get("conversation_id", ""))
        annotations = item.get("annotations", [])

        if not conv_id:
            errors.append("Missing conversation_id in one item")
            continue

        if expected_ids and conv_id not in expected_ids:
            errors.append(f"Unexpected conversation_id: {conv_id}")
            continue

        if conv_id in completed_ids:
            skipped += 1
            continue

        # Validate labels
        valid_labels = {"essential", "redundant", "uncertain"}
        for ann in annotations:
            if ann.get("label") not in valid_labels:
                ann["label"] = "uncertain"  # coerce bad labels

        # Compute stats
        total = len(annotations)
        essential = sum(1 for a in annotations if a["label"] == "essential")
        redundant = sum(1 for a in annotations if a["label"] == "redundant")
        uncertain = sum(1 for a in annotations if a["label"] == "uncertain")
        scr = redundant / max(1, total)

        record = {
            "conversation_id": conv_id,
            "total_sentences": total,
            "essential": essential,
            "redundant": redundant,
            "uncertain": uncertain,
            "safe_compression_rate": round(scr, 4),
            "annotations": annotations,
            "annotator": "gpt-5.2-pro-manual",
            "sentences": batch_sentences.get(conv_id, []),
        }

        with open(ANNOTATIONS_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed_ids.add(conv_id)
        progress["completed_ids"].append(conv_id)
        saved += 1

        print(f"  ✅ {conv_id}: {total} sentences, "
              f"ess={essential} red={redundant} unc={uncertain} "
              f"scr={scr:.1%}")

    _save_progress(progress)

    # Clear pending batch
    if pending_path.exists() and not errors:
        pending_path.unlink()

    print(f"\nSaved: {saved} | Skipped (already done): {skipped}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  ⚠️  {e}")

    # Update stats
    _update_stats()


def cmd_status(args) -> None:
    """Show annotation progress."""
    conversations = _load_sharegpt()
    total = len(conversations)
    annotations = _load_annotations()
    done = len(annotations)
    remaining = total - done

    print(f"\n=== CompressionSafety Annotation Progress ===")
    print(f"  Total conversations: {total}")
    print(f"  Annotated:          {done} ({done/total:.1%})")
    print(f"  Remaining:          {remaining}")

    if annotations:
        scr_values = [a["safe_compression_rate"] for a in annotations]
        mean_scr = sum(scr_values) / len(scr_values)
        var = sum((x - mean_scr) ** 2 for x in scr_values) / len(scr_values)
        std = math.sqrt(var)
        print(f"\n  Safe compression rate so far:")
        print(f"    Mean: {mean_scr:.1%}")
        print(f"    Std:  {std:.1%}")
        print(f"    Min:  {min(scr_values):.1%}")
        print(f"    Max:  {max(scr_values):.1%}")

        annotators = {}
        for a in annotations:
            ann = a.get("annotator", "unknown")
            annotators[ann] = annotators.get(ann, 0) + 1
        print(f"\n  By annotator:")
        for ann, count in annotators.items():
            print(f"    {ann}: {count}")

    batches_needed = math.ceil(remaining / BATCH_SIZE_DEFAULT)
    print(f"\n  Estimated batches remaining (size={BATCH_SIZE_DEFAULT}): {batches_needed}")
    print(f"  Estimated ChatGPT time: ~{batches_needed * 2} minutes")
    print()


def _update_stats() -> None:
    annotations = _load_annotations()
    if not annotations:
        return

    scr_values = [a["safe_compression_rate"] for a in annotations]
    mean_scr = sum(scr_values) / len(scr_values)
    var = sum((x - mean_scr) ** 2 for x in scr_values) / len(scr_values)
    std = math.sqrt(var)

    stats = {
        "n_annotated": len(annotations),
        "mean_safe_compression_rate": round(mean_scr, 4),
        "std_safe_compression_rate": round(std, 4),
        "min_scr": round(min(scr_values), 4),
        "max_scr": round(max(scr_values), 4),
        "annotator": "gpt-5.2-pro-manual",
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)


def cmd_export(args) -> None:
    """Export cleaned dataset.jsonl ready for HuggingFace."""
    annotations = _load_annotations()
    if not annotations:
        print("No annotations yet.")
        return

    dataset_path = OUT_DIR / "dataset.jsonl"
    with open(dataset_path, "w") as f:
        for a in annotations:
            record = {
                "conversation_id": a["conversation_id"],
                "total_sentences": a["total_sentences"],
                "essential": a["essential"],
                "redundant": a["redundant"],
                "uncertain": a.get("uncertain", 0),
                "safe_compression_rate": a["safe_compression_rate"],
                "sentences": a.get("sentences", []),
                "annotations": a["annotations"],
                "annotator": a.get("annotator", "unknown"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Exported {len(annotations)} annotations → {dataset_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manual CompressionSafety annotation using ChatGPT Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_next = sub.add_parser("next-batch", help="Get next batch prompt to paste into ChatGPT")
    p_next.add_argument("--size", type=int, default=BATCH_SIZE_DEFAULT,
                        help=f"Conversations per batch (default: {BATCH_SIZE_DEFAULT})")
    p_next.set_defaults(func=cmd_next_batch)

    p_ingest = sub.add_parser("ingest", help="Ingest ChatGPT's JSON response")
    p_ingest.add_argument("file", help="Path to file containing ChatGPT's JSON response")
    p_ingest.set_defaults(func=cmd_ingest)

    p_status = sub.add_parser("status", help="Show annotation progress")
    p_status.set_defaults(func=cmd_status)

    p_export = sub.add_parser("export", help="Export cleaned dataset.jsonl for HuggingFace")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
