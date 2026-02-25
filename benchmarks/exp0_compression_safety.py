"""
Experiment 0: Build the CompressionSafety dataset.

Annotates real ShareGPT conversations sentence-by-sentence.
Labels each sentence as essential (1) or redundant (0).

Cost estimate: ~$8 total for 500 conversations via gpt-5.3-codex.

Usage:
  python3 benchmarks/exp0_compression_safety.py --n 500 --annotator codex --dry-run
  python3 benchmarks/exp0_compression_safety.py --n 500 --annotator codex
  python3 benchmarks/exp0_compression_safety.py --n 50 --annotator claude --budget 2.00

Output:
  benchmarks/data/compression_safety/annotations.jsonl  — raw per-sentence annotations
  benchmarks/data/compression_safety/dataset.jsonl       — cleaned dataset (HuggingFace-ready)
  benchmarks/data/compression_safety/stats.json          — summary stats
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextprune.adapters.openrouter import SUPPORTED_MODELS, OpenRouterAdapter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
SHAREGPT_PATH = DATA_DIR / "sharegpt" / "test.jsonl"
OUT_DIR = DATA_DIR / "compression_safety"
OUT_DIR.mkdir(exist_ok=True)

ANNOTATIONS_PATH = OUT_DIR / "annotations.jsonl"
DATASET_PATH = OUT_DIR / "dataset.jsonl"
STATS_PATH = OUT_DIR / "stats.json"

# ---------------------------------------------------------------------------
# Annotation prompt
# ---------------------------------------------------------------------------

ANNOTATION_PROMPT = """\
You are annotating a conversation for a compression research study.

Given this conversation and its final question, classify each sentence as:
- essential: removing this sentence would change the correct answer
- redundant: removing this sentence would NOT change the correct answer
- uncertain: unclear whether removal would affect the answer

Conversation:
{conversation}

Final question:
{question}

Sentences to annotate:
{sentences_json}

Return a JSON array with one object per sentence:
[{{"sentence_id": 0, "text": "...", "label": "essential|redundant|uncertain", "reason": "brief reason"}}]

Return ONLY the JSON array, no preamble or explanation.
"""

# ---------------------------------------------------------------------------
# Model aliases for annotators
# ---------------------------------------------------------------------------

ANNOTATOR_ALIASES = {
    "codex": "codex",     # gpt-5.3-codex
    "claude": "claude",   # claude-sonnet-4-6
    "gpt52": "gpt52",     # gpt-5.2
    "grok": "grok",       # grok-4.1-fast (cheapest option)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sharegpt(n: int, seed: int = 42) -> list[dict]:
    """Load n ShareGPT conversations from the downloaded dataset."""
    if not SHAREGPT_PATH.exists():
        raise FileNotFoundError(
            f"ShareGPT dataset not found at {SHAREGPT_PATH}.\n"
            f"Run: python3 benchmarks/data/download_datasets.py --dataset sharegpt"
        )
    items: list[dict] = []
    with open(SHAREGPT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def _conversation_to_text(conversations: list[dict]) -> str:
    """Convert conversation turns to a flat text string."""
    parts = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from", turn.get("role", "human"))
        value = turn.get("value", turn.get("content", "")).strip()
        if value:
            role_label = "Human" if role in ("human", "user") else "Assistant"
            parts.append(f"{role_label}: {value}")
    return "\n\n".join(parts)


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences for annotation.
    Handles abbreviations, decimal points, etc.
    Returns list of non-empty sentence strings.
    """
    # Simple sentence splitter — good enough for annotation
    # Split on . ? ! followed by whitespace and capital letter, or newlines
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+|(?<=\n)\n+", text)
    # Further split on very long sentences (>200 chars) at commas
    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > 400:
            # Split at midpoint comma
            parts = sent.split(", ")
            chunk = ""
            for part in parts:
                if len(chunk) + len(part) < 200:
                    chunk = (chunk + ", " + part).lstrip(", ")
                else:
                    if chunk:
                        result.append(chunk)
                    chunk = part
            if chunk:
                result.append(chunk)
        else:
            result.append(sent)
    return [s for s in result if len(s) > 5]  # skip very short fragments


def _parse_annotations(raw_text: str, sentences: list[str]) -> list[dict]:
    """
    Parse model's JSON annotation response.
    Falls back to marking all as 'uncertain' if parsing fails.
    """
    # Extract JSON array
    json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not json_match:
        return [
            {"sentence_id": i, "text": s, "label": "uncertain", "reason": "parse_error"}
            for i, s in enumerate(sentences)
        ]
    try:
        parsed = json.loads(json_match.group(0))
        if not isinstance(parsed, list):
            raise ValueError("Not a list")
        # Validate and normalize labels
        valid_labels = {"essential", "redundant", "uncertain"}
        result = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            label = item.get("label", "uncertain").lower()
            if label not in valid_labels:
                label = "uncertain"
            result.append({
                "sentence_id": int(item.get("sentence_id", 0)),
                "text": str(item.get("text", "")),
                "label": label,
                "reason": str(item.get("reason", "")),
            })
        # Fill in any missing sentence IDs
        found_ids = {r["sentence_id"] for r in result}
        for i, s in enumerate(sentences):
            if i not in found_ids:
                result.append({"sentence_id": i, "text": s, "label": "uncertain", "reason": "missing"})
        result.sort(key=lambda x: x["sentence_id"])
        return result
    except Exception:
        return [
            {"sentence_id": i, "text": s, "label": "uncertain", "reason": "parse_error"}
            for i, s in enumerate(sentences)
        ]


def _load_completed_ids() -> set[str]:
    """Return set of conversation IDs already annotated."""
    completed: set[str] = set()
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    completed.add(str(item["conversation_id"]))
                except Exception:
                    pass
    return completed


def _load_openrouter_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    try:
        result = subprocess.run(
            ["bash", "-c",
             "OP_SERVICE_ACCOUNT_TOKEN=$(cat /etc/op-service-account-token) "
             "op item get 'OPENROUTER_API_KEY' --vault 'Keef Secrets' "
             "--fields credential --reveal 2>/dev/null"],
            capture_output=True, text=True, timeout=15,
        )
        key = result.stdout.strip()
        if key and key not in ("None", "", "null"):
            return key
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main annotation runner
# ---------------------------------------------------------------------------

def run_annotation(
    n: int = 500,
    annotator: str = "codex",
    budget_usd: float = 10.0,
    dry_run: bool = False,
    seed: int = 42,
    max_sentences_per_conv: int = 30,
) -> dict:
    """
    Annotate n ShareGPT conversations.

    Returns summary stats dict.
    """
    print(f"\n=== Experiment 0: CompressionSafety Annotation ===")
    print(f"  n={n}, annotator={annotator}, budget=${budget_usd:.2f}, dry_run={dry_run}")

    # Load conversations
    conversations = _load_sharegpt(n, seed=seed)
    print(f"  Loaded {len(conversations):,} conversations from ShareGPT")

    # Check which are already done
    completed_ids = _load_completed_ids()
    remaining = [c for c in conversations if str(c.get("id", "")) not in completed_ids]
    print(f"  Already annotated: {len(completed_ids):,}. Remaining: {len(remaining):,}")

    if dry_run:
        # Show preview of what would be annotated
        print(f"\n  [DRY RUN] Would annotate {len(remaining)} conversations")
        for i, conv in enumerate(remaining[:3]):
            conv_text = _conversation_to_text(conv.get("conversations", []))
            sentences = _split_into_sentences(conv_text)
            sentences = sentences[:max_sentences_per_conv]
            print(f"\n  Conv {i+1}: {len(sentences)} sentences, ~{len(conv_text)//4} tokens")
            print(f"    First sentence: {sentences[0][:80]!r}…" if sentences else "    (empty)")
            print(f"    Last sentence:  {sentences[-1][:80]!r}…" if sentences else "")
        # Estimate cost
        n_sentences_est = n * 15  # avg 15 sentences per conversation
        avg_tokens_per_call = 800  # prompt + sentences
        total_calls = len(remaining)
        est_cost = total_calls * (avg_tokens_per_call / 1e6) * 1.75 * 4  # ~$0.007/call
        print(f"\n  Estimated total cost: ${est_cost:.2f}")
        print(f"  Estimated total sentences: {n_sentences_est:,}")
        return {"dry_run": True, "n_would_annotate": len(remaining), "est_cost": est_cost}

    # Real annotation
    api_key = _load_openrouter_key()
    if not api_key:
        print("  [ERROR] No OPENROUTER_API_KEY found. Cannot annotate.")
        return {"error": "no_api_key"}

    adapter = OpenRouterAdapter(api_key=api_key)
    annotator_alias = ANNOTATOR_ALIASES.get(annotator, annotator)

    spent = 0.0
    all_stats: list[dict] = []

    for conv_idx, conv in enumerate(remaining):
        if spent >= budget_usd:
            print(f"\n  [BUDGET] Reached ${budget_usd:.2f}. Stopping at {conv_idx}/{len(remaining)}.")
            break

        conv_id = str(conv.get("id", f"conv_{conv_idx:05d}"))
        conversations_raw = conv.get("conversations", [])
        conv_text = _conversation_to_text(conversations_raw)
        final_q = conv.get("question", "")
        if not final_q and conversations_raw:
            human_turns = [c for c in conversations_raw if isinstance(c, dict)
                           and c.get("from", c.get("role", "")) in ("human", "user")]
            final_q = human_turns[-1].get("value", human_turns[-1].get("content", "")) if human_turns else ""

        sentences = _split_into_sentences(conv_text)[:max_sentences_per_conv]
        if not sentences:
            continue

        sentences_json = json.dumps([{"sentence_id": i, "text": s} for i, s in enumerate(sentences)])

        prompt = ANNOTATION_PROMPT.format(
            conversation=conv_text[:3000],  # cap context
            question=final_q[:500],
            sentences_json=sentences_json,
        )

        t0 = time.perf_counter()
        try:
            result = adapter.complete(
                messages=[{"role": "user", "content": prompt}],
                model=annotator_alias,
                max_tokens=2048,
                temperature=0.1,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            spent += result.cost_usd
            annotations = _parse_annotations(result.text, sentences)

            # Compute stats for this conversation
            total_sentences = len(annotations)
            essential_count = sum(1 for a in annotations if a["label"] == "essential")
            redundant_count = sum(1 for a in annotations if a["label"] == "redundant")
            uncertain_count = sum(1 for a in annotations if a["label"] == "uncertain")
            safe_compression_rate = redundant_count / max(1, total_sentences)

            annotated_record = {
                "conversation_id": conv_id,
                "total_sentences": total_sentences,
                "essential": essential_count,
                "redundant": redundant_count,
                "uncertain": uncertain_count,
                "safe_compression_rate": round(safe_compression_rate, 4),
                "annotations": annotations,
                "annotator": annotator_alias,
                "cost_usd": result.cost_usd,
                "latency_ms": round(latency_ms, 2),
            }

            # Write raw annotation
            with open(ANNOTATIONS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(annotated_record, ensure_ascii=False) + "\n")

            all_stats.append({
                "conversation_id": conv_id,
                "total_sentences": total_sentences,
                "safe_compression_rate": safe_compression_rate,
            })

            print(f"  Conv {conv_idx+1:04d}/{len(remaining)}: "
                  f"{total_sentences}s  "
                  f"ess={essential_count} red={redundant_count} unc={uncertain_count} "
                  f"scr={safe_compression_rate:.1%}  "
                  f"${result.cost_usd:.5f}  total=${spent:.4f}")

        except Exception as e:
            print(f"  [ERROR] {conv_id}: {e}")
            continue

    # Build cleaned dataset.jsonl
    _build_cleaned_dataset()

    # Compute summary stats
    if all_stats:
        scr_values = [s["safe_compression_rate"] for s in all_stats]
        import math
        mean_scr = sum(scr_values) / len(scr_values)
        var_scr = sum((x - mean_scr) ** 2 for x in scr_values) / max(1, len(scr_values))
        std_scr = math.sqrt(var_scr)
        summary = {
            "n_annotated": len(all_stats),
            "mean_safe_compression_rate": round(mean_scr, 4),
            "std_safe_compression_rate": round(std_scr, 4),
            "min_scr": round(min(scr_values), 4),
            "max_scr": round(max(scr_values), 4),
            "total_cost_usd": round(spent, 6),
            "annotator": annotator,
        }
        with open(STATS_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== CompressionSafety Stats ===")
        print(f"  Annotated: {summary['n_annotated']:,} conversations")
        print(f"  Safe compression rate: {mean_scr:.1%} ± {std_scr:.1%}")
        print(f"  Total cost: ${spent:.4f}")
        print(f"  Annotations → {ANNOTATIONS_PATH}")
        print(f"  Dataset     → {DATASET_PATH}")
        print(f"  Stats       → {STATS_PATH}")
        return summary
    return {"n_annotated": 0}


def _build_cleaned_dataset() -> None:
    """Build cleaned dataset.jsonl from raw annotations."""
    if not ANNOTATIONS_PATH.exists():
        return

    records: list[dict] = []
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                record = {
                    "conversation_id": raw["conversation_id"],
                    "total_sentences": raw["total_sentences"],
                    "essential_count": raw["essential"],
                    "redundant_count": raw["redundant"],
                    "uncertain_count": raw["uncertain"],
                    "safe_compression_rate": raw["safe_compression_rate"],
                    "sentences": [
                        {
                            "id": a["sentence_id"],
                            "text": a["text"],
                            "label": a["label"],
                            "is_essential": 1 if a["label"] == "essential" else 0,
                            "is_redundant": 1 if a["label"] == "redundant" else 0,
                        }
                        for a in raw.get("annotations", [])
                    ],
                }
                records.append(record)
            except Exception:
                pass

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Built cleaned dataset: {len(records):,} records → {DATASET_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 0: CompressionSafety annotation.")
    p.add_argument("--n", type=int, default=500, help="Number of conversations to annotate")
    p.add_argument("--annotator", default="codex",
                   choices=list(ANNOTATOR_ALIASES.keys()),
                   help="Model to use for annotation")
    p.add_argument("--budget", type=float, default=10.0, help="Budget in USD")
    p.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sentences", type=int, default=30,
                   help="Max sentences per conversation to annotate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_annotation(
        n=args.n,
        annotator=args.annotator,
        budget_usd=args.budget,
        dry_run=args.dry_run,
        seed=args.seed,
        max_sentences_per_conv=args.max_sentences,
    )
    if not args.dry_run and "mean_safe_compression_rate" in result:
        print(f"\nKey result: safe_compression_rate = {result['mean_safe_compression_rate']:.1%} ± {result['std_safe_compression_rate']:.1%}")


if __name__ == "__main__":
    main()
