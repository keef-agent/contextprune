#!/usr/bin/env python3
"""
CompressionSafety Verification Suite

Three verification layers for the CompressionSafety dataset:

  Layer 1 — Second-annotator IAA (Claude Sonnet 4.6 via OpenRouter)
    Resample 30 conversations (stratified by SCR bucket), re-annotate with Claude,
    compute Cohen's Kappa vs GPT-5.2 labels.

  Layer 2 — Compression-then-test (functional validation)
    Pick 20 conversations, build compressed (essential-only) context, ask Claude
    the final question with both full and compressed context, compare answers via
    cosine similarity.

  Layer 3 — Human spot-check report
    Generate a readable Markdown file with 25 random sentence-level decisions
    formatted for human review.

CLI:
  python3 benchmarks/compression_safety_verify.py iaa --n 30
  python3 benchmarks/compression_safety_verify.py functional --n 20
  python3 benchmarks/compression_safety_verify.py human-review --n 25
  python3 benchmarks/compression_safety_verify.py all --iaa-n 30 --func-n 20 --review-n 25
  python3 benchmarks/compression_safety_verify.py all --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
OUT_DIR = REPO_ROOT / "benchmarks" / "data" / "compression_safety"
ANNOTATIONS_PATH = OUT_DIR / "annotations.jsonl"
IAA_RESULTS_PATH = OUT_DIR / "iaa_results.json"
FUNCTIONAL_RESULTS_PATH = OUT_DIR / "functional_results.json"
HUMAN_REVIEW_PATH = OUT_DIR / "human_review.md"

SEED = 42
CLAUDE_MODEL = "anthropic/claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Cost estimates (per million tokens)
# ---------------------------------------------------------------------------
CLAUDE_INPUT_PRICE  = 3.00   # $/1M tokens
CLAUDE_OUTPUT_PRICE = 15.00  # $/1M tokens

# ---------------------------------------------------------------------------
# Annotation prompt — identical to compression_safety_manual.py
# ---------------------------------------------------------------------------

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
        conv_id = item["conversation_id"]
        sentences = item.get("sentences", [])
        final_q = item.get("final_q", "")

        sentences_formatted = "\n".join(
            f"  [{i}] {s}" for i, s in enumerate(sentences)
        )

        block = f"""=== CONVERSATION {conv_id} ===
Sentences to label:
{sentences_formatted}

Final question being answered:
{final_q[:300] or "(see last sentence above)"}"""
        blocks.append(block)

    return "\n\n" + ("-" * 60 + "\n\n").join(blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_annotations() -> list[dict]:
    if not ANNOTATIONS_PATH.exists():
        return []
    with open(ANNOTATIONS_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def _get_scr_bucket(scr: float) -> str:
    """Return bucket label for SCR value 0.0–1.0."""
    if scr < 0.20:
        return "0-20"
    elif scr < 0.40:
        return "20-40"
    elif scr < 0.60:
        return "40-60"
    elif scr < 0.80:
        return "60-80"
    else:
        return "80-100"


def _stratified_sample(annotations: list[dict], n_per_bucket: int, seed: int = SEED) -> list[dict]:
    """Sample n_per_bucket conversations from each SCR bucket, fill from leftovers if needed."""
    rng = random.Random(seed)

    buckets: dict[str, list[dict]] = {
        "0-20": [], "20-40": [], "40-60": [], "60-80": [], "80-100": []
    }
    for ann in annotations:
        scr = ann.get("safe_compression_rate", 0.5)
        bucket = _get_scr_bucket(scr)
        buckets[bucket].append(ann)

    sampled: list[dict] = []
    leftover: list[dict] = []

    for bucket_name, items in buckets.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:n_per_bucket])
        leftover.extend(shuffled[n_per_bucket:])

    # If any bucket was short, fill from leftover
    total_needed = n_per_bucket * len(buckets)
    if len(sampled) < total_needed:
        rng.shuffle(leftover)
        sampled.extend(leftover[:total_needed - len(sampled)])

    return sampled


def _load_openrouter_key() -> str:
    """Load OpenRouter API key from env or 1Password."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    print("ℹ️  OPENROUTER_API_KEY not set — attempting to fetch from 1Password...", file=sys.stderr)
    try:
        import subprocess
        sa_token = open("/etc/op-service-account-token").read().strip()
        result = subprocess.run(
            ["op", "item", "get", "OPENROUTER_API_KEY", "--vault", "Keef Secrets",
             "--fields", "credential", "--reveal"],
            capture_output=True, text=True,
            env={**os.environ, "OP_SERVICE_ACCOUNT_TOKEN": sa_token},
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"⚠️  1Password fetch failed: {e}", file=sys.stderr)
    print("❌ Could not load OPENROUTER_API_KEY. Set it as env var or ensure 1Password CLI is available.", file=sys.stderr)
    sys.exit(1)


def _make_adapter(api_key: str):
    """Create an OpenRouterAdapter."""
    sys.path.insert(0, str(REPO_ROOT))
    from contextprune.adapters.openrouter import OpenRouterAdapter
    return OpenRouterAdapter(api_key=api_key)


def _parse_annotation_json(text: str) -> list[dict]:
    """Parse Claude's JSON response, stripping markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------

def _compute_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """
    Compute Cohen's Kappa for binary labels (essential=0, redundant=1).
    Uncertain labels are excluded from the calculation.
    κ = (p_o - p_e) / (1 - p_e)
    """
    assert len(labels_a) == len(labels_b)

    pairs = [
        (a, b) for a, b in zip(labels_a, labels_b)
        if a != "uncertain" and b != "uncertain"
    ]

    if not pairs:
        return float("nan")

    n = len(pairs)
    # Map labels to binary: essential=0, redundant=1
    def _to_bin(lbl: str) -> int:
        return 0 if lbl == "essential" else 1

    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n

    # Marginal proportions
    p_a_red = sum(1 for a, _ in pairs if a == "redundant") / n
    p_b_red = sum(1 for _, b in pairs if b == "redundant") / n
    p_a_ess = 1 - p_a_red
    p_b_ess = 1 - p_b_red

    p_e = p_a_ess * p_b_ess + p_a_red * p_b_red

    if p_e >= 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1 - p_e)
    return round(kappa, 4)


def _interpret_kappa(k: float) -> str:
    if math.isnan(k):
        return "N/A (no comparable pairs)"
    if k < 0.4:
        return "poor"
    elif k < 0.6:
        return "moderate"
    elif k < 0.8:
        return "substantial"
    else:
        return "almost perfect"


# ---------------------------------------------------------------------------
# Cosine similarity (no external deps — pure Python TF-IDF style embedding)
# ---------------------------------------------------------------------------

def _simple_embed(text: str) -> dict[str, float]:
    """Very simple bag-of-words TF vector (lowercase unigrams)."""
    words = re.findall(r"[a-z]+", text.lower())
    vec: dict[str, float] = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        return {k: v / norm for k, v in vec.items()}
    return vec


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Dot product of two L2-normalized sparse vectors."""
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a)
    return round(dot, 4)


# ---------------------------------------------------------------------------
# Layer 1: IAA
# ---------------------------------------------------------------------------

def run_iaa(n: int = 30, api_key: str | None = None, dry_run: bool = False) -> dict:
    """
    Second-annotator IAA: re-annotate n conversations with Claude Sonnet 4.6,
    compute Cohen's Kappa vs GPT-5.2 labels.
    """
    print(f"\n{'='*60}")
    print(f"Layer 1: IAA — {n} conversations, model={CLAUDE_MODEL}")
    print(f"{'='*60}")

    annotations = _load_annotations()
    if not annotations:
        print("❌ No annotations found. Run compression_safety_manual.py first.")
        sys.exit(1)

    n_per_bucket = n // 5  # 5 SCR buckets
    sample = _stratified_sample(annotations, n_per_bucket)[:n]

    if dry_run:
        est_input  = n * 800
        est_output = n * 600
        est_cost   = (est_input / 1e6) * CLAUDE_INPUT_PRICE + (est_output / 1e6) * CLAUDE_OUTPUT_PRICE
        print(f"\n💰 Dry-run cost estimate (Layer 1):")
        print(f"   {n} calls × ~800 tokens input  = {est_input:,} tokens → ${est_input/1e6*CLAUDE_INPUT_PRICE:.3f}")
        print(f"   {n} calls × ~600 tokens output = {est_output:,} tokens → ${est_output/1e6*CLAUDE_OUTPUT_PRICE:.3f}")
        print(f"   Total Layer 1 estimate: ~${est_cost:.3f}")
        return {"dry_run": True, "estimated_cost_usd": round(est_cost, 3)}

    if not api_key:
        api_key = _load_openrouter_key()
    adapter = _make_adapter(api_key)

    # Load existing IAA results for checkpointing
    existing: dict[str, dict] = {}
    if IAA_RESULTS_PATH.exists():
        with open(IAA_RESULTS_PATH) as f:
            existing = json.load(f)
        print(f"  ↩️  Resuming — {len(existing)} already annotated")

    claude_annotations: dict[str, list[dict]] = dict(existing)
    total_cost = 0.0

    for idx, ann in enumerate(sample, 1):
        conv_id = ann["conversation_id"]
        if conv_id in claude_annotations:
            print(f"  [{idx:2}/{len(sample)}] {conv_id} — skipped (already done)")
            continue

        sentences = ann.get("sentences", [])
        if not sentences:
            print(f"  [{idx:2}/{len(sample)}] {conv_id} — no sentences, skipping")
            continue

        # Infer final_q from last sentence if not stored
        final_q = ""
        if sentences:
            human_turns = [s for s in sentences if s.startswith("User:")]
            final_q = human_turns[-1] if human_turns else sentences[-1]

        item = {
            "conversation_id": conv_id,
            "sentences": sentences,
            "final_q": final_q[:300],
        }

        prompt = ANNOTATION_PROMPT_TEMPLATE.format(
            conversations_block=_format_conversation_block([item])
        )

        try:
            result = adapter.complete(
                messages=[{"role": "user", "content": prompt}],
                model=CLAUDE_MODEL,
                max_tokens=1500,
                temperature=0.0,
            )
            total_cost += result.cost_usd

            parsed = _parse_annotation_json(result.text)
            if parsed and isinstance(parsed, list):
                claude_anns = parsed[0].get("annotations", [])
            else:
                claude_anns = []

            claude_annotations[conv_id] = claude_anns
            print(f"  [{idx:2}/{len(sample)}] {conv_id} — {len(claude_anns)} labels, cost=${result.cost_usd:.4f}")

        except Exception as e:
            print(f"  [{idx:2}/{len(sample)}] {conv_id} — ERROR: {e}")
            claude_annotations[conv_id] = []

        # Checkpoint after each annotation
        with open(IAA_RESULTS_PATH, "w") as f:
            json.dump(claude_annotations, f, indent=2)

    # Compute per-conversation kappa
    print(f"\n  Computing Cohen's Kappa...")
    kappa_scores: list[float] = []
    bucket_kappas: dict[str, list[float]] = {k: [] for k in ["0-20", "20-40", "40-60", "60-80", "80-100"]}
    disagree_essential = 0   # Claude says essential, GPT says redundant
    disagree_redundant = 0   # Claude says redundant, GPT says essential
    total_pairs = 0

    conv_results = []
    for ann in sample:
        conv_id = ann["conversation_id"]
        gpt_anns = {a["sentence_id"]: a["label"] for a in ann.get("annotations", [])}
        claude_anns_raw = claude_annotations.get(conv_id, [])
        claude_anns_map = {a["sentence_id"]: a["label"] for a in claude_anns_raw}

        # Align on sentence IDs present in both
        common_ids = sorted(set(gpt_anns) & set(claude_anns_map))
        if not common_ids:
            continue

        gpt_labels   = [gpt_anns[i] for i in common_ids]
        claude_labels = [claude_anns_map[i] for i in common_ids]

        k = _compute_kappa(gpt_labels, claude_labels)
        kappa_scores.append(k)

        bucket = _get_scr_bucket(ann["safe_compression_rate"])
        bucket_kappas[bucket].append(k)

        # Disagreement counting (exclude uncertain)
        for g, c in zip(gpt_labels, claude_labels):
            if g == "uncertain" or c == "uncertain":
                continue
            total_pairs += 1
            if g != c:
                if g == "essential" and c == "redundant":
                    disagree_essential += 1
                elif g == "redundant" and c == "essential":
                    disagree_redundant += 1

        conv_results.append({
            "conversation_id": conv_id,
            "scr": ann["safe_compression_rate"],
            "bucket": bucket,
            "kappa": k,
            "n_pairs": len([g for g, c in zip(gpt_labels, claude_labels) if g != "uncertain" and c != "uncertain"]),
        })

    # Aggregate
    valid_kappas = [k for k in kappa_scores if not math.isnan(k)]
    overall_kappa = sum(valid_kappas) / len(valid_kappas) if valid_kappas else float("nan")

    per_bucket = {}
    for bucket, ks in bucket_kappas.items():
        valid = [k for k in ks if not math.isnan(k)]
        per_bucket[bucket] = round(sum(valid) / len(valid), 4) if valid else float("nan")

    disagree_rate_essential = disagree_essential / max(1, total_pairs)
    disagree_rate_redundant = disagree_redundant / max(1, total_pairs)

    summary = {
        "overall_kappa": round(overall_kappa, 4),
        "interpretation": _interpret_kappa(overall_kappa),
        "per_bucket_kappa": per_bucket,
        "disagree_rate_essential_labeled_redundant": round(disagree_rate_essential, 4),
        "disagree_rate_redundant_labeled_essential": round(disagree_rate_redundant, 4),
        "total_comparable_pairs": total_pairs,
        "n_conversations": len(conv_results),
        "total_cost_usd": round(total_cost, 4),
        "conversations": conv_results,
    }

    # Save full results
    with open(IAA_RESULTS_PATH, "w") as f:
        json.dump({
            "summary": summary,
            "claude_annotations": claude_annotations,
        }, f, indent=2)

    # Print report
    print(f"\n{'='*60}")
    print(f"  Layer 1 Results — Cohen's Kappa")
    print(f"{'='*60}")
    print(f"  Overall κ = {overall_kappa:.4f}  ({_interpret_kappa(overall_kappa)})")
    print(f"\n  Per-bucket κ:")
    for bucket, k in per_bucket.items():
        print(f"    SCR {bucket}%: κ = {k:.4f}  ({_interpret_kappa(k)})")
    print(f"\n  Disagreement rates (of {total_pairs} comparable sentence pairs):")
    print(f"    GPT=essential, Claude=redundant: {disagree_rate_essential:.1%}  ({disagree_essential} pairs)")
    print(f"    GPT=redundant, Claude=essential: {disagree_rate_redundant:.1%}  ({disagree_redundant} pairs)")
    print(f"\n  Total API cost: ${total_cost:.4f}")
    print(f"  Results saved → {IAA_RESULTS_PATH}")

    return summary


# ---------------------------------------------------------------------------
# Layer 2: Functional validation
# ---------------------------------------------------------------------------

def run_functional(n: int = 20, api_key: str | None = None, dry_run: bool = False) -> dict:
    """
    Compression-then-test: ask Claude the final question with full vs compressed context,
    compare answers via cosine similarity.
    """
    print(f"\n{'='*60}")
    print(f"Layer 2: Functional — {n} conversations, model={CLAUDE_MODEL}")
    print(f"{'='*60}")

    annotations = _load_annotations()
    if not annotations:
        print("❌ No annotations found.")
        sys.exit(1)

    # Prefer conversations that have sentences (needed to build compressed context)
    valid = [a for a in annotations if a.get("sentences")]
    rng = random.Random(SEED + 1)
    rng.shuffle(valid)
    sample = valid[:n]

    if dry_run:
        est_input  = n * 2 * 500  # 2 calls per conversation
        est_output = n * 2 * 300
        est_cost   = (est_input / 1e6) * CLAUDE_INPUT_PRICE + (est_output / 1e6) * CLAUDE_OUTPUT_PRICE
        print(f"\n💰 Dry-run cost estimate (Layer 2):")
        print(f"   {n} convs × 2 calls × ~500 tokens input  = {est_input:,} tokens → ${est_input/1e6*CLAUDE_INPUT_PRICE:.3f}")
        print(f"   {n} convs × 2 calls × ~300 tokens output = {est_output:,} tokens → ${est_output/1e6*CLAUDE_OUTPUT_PRICE:.3f}")
        print(f"   Total Layer 2 estimate: ~${est_cost:.3f}")
        return {"dry_run": True, "estimated_cost_usd": round(est_cost, 3)}

    if not api_key:
        api_key = _load_openrouter_key()
    adapter = _make_adapter(api_key)

    # Load existing results for checkpointing
    existing_results: list[dict] = []
    existing_ids: set[str] = set()
    if FUNCTIONAL_RESULTS_PATH.exists():
        with open(FUNCTIONAL_RESULTS_PATH) as f:
            data = json.load(f)
            existing_results = data.get("results", [])
            existing_ids = {r["conversation_id"] for r in existing_results}
        print(f"  ↩️  Resuming — {len(existing_ids)} already done")

    results: list[dict] = list(existing_results)
    total_cost = 0.0
    SIMILARITY_THRESHOLD = 0.85

    for idx, ann in enumerate(sample, 1):
        conv_id = ann["conversation_id"]
        if conv_id in existing_ids:
            print(f"  [{idx:2}/{len(sample)}] {conv_id} — skipped (already done)")
            continue

        sentences = ann.get("sentences", [])
        annotations_list = ann.get("annotations", [])

        # Build essential-only (compressed) context
        essential_ids = {
            a["sentence_id"] for a in annotations_list
            if a.get("label") == "essential"
        }
        full_context    = "\n".join(sentences)
        compressed_ctx  = "\n".join(s for i, s in enumerate(sentences) if i in essential_ids)

        # Extract final question = last User: turn
        human_turns = [s for s in sentences if s.startswith("User:")]
        final_q = human_turns[-1] if human_turns else (sentences[-1] if sentences else "What is the answer?")
        final_q_text = final_q.replace("User:", "").strip()[:500]

        full_tokens      = len(full_context.split())
        compressed_tokens = len(compressed_ctx.split())
        reduction_pct = (full_tokens - compressed_tokens) / max(1, full_tokens)

        def ask_claude(context: str, question: str) -> tuple[str, float]:
            prompt = f"""Given the following conversation context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""
            r = adapter.complete(
                messages=[{"role": "user", "content": prompt}],
                model=CLAUDE_MODEL,
                max_tokens=500,
                temperature=0.0,
            )
            return r.text, r.cost_usd

        try:
            answer_full,       cost_full       = ask_claude(full_context,    final_q_text)
            answer_compressed, cost_compressed = ask_claude(compressed_ctx,  final_q_text)
            total_cost += cost_full + cost_compressed

            sim = _cosine_similarity(
                _simple_embed(answer_full),
                _simple_embed(answer_compressed),
            )
            preserved = sim >= SIMILARITY_THRESHOLD

            entry = {
                "conversation_id": conv_id,
                "scr": ann["safe_compression_rate"],
                "full_tokens": full_tokens,
                "compressed_tokens": compressed_tokens,
                "token_reduction_pct": round(reduction_pct, 4),
                "cosine_similarity": sim,
                "answer_preserved": preserved,
                "answer_full": answer_full[:800],
                "answer_compressed": answer_compressed[:800],
                "cost_usd": round(cost_full + cost_compressed, 5),
            }
            results.append(entry)
            existing_ids.add(conv_id)

            status = "✅ preserved" if preserved else "❌ changed"
            print(f"  [{idx:2}/{len(sample)}] {conv_id} — sim={sim:.3f} {status}  "
                  f"reduction={reduction_pct:.0%}  cost=${entry['cost_usd']:.4f}")

        except Exception as e:
            print(f"  [{idx:2}/{len(sample)}] {conv_id} — ERROR: {e}")
            results.append({
                "conversation_id": conv_id,
                "error": str(e),
            })
            existing_ids.add(conv_id)

        # Checkpoint
        with open(FUNCTIONAL_RESULTS_PATH, "w") as f:
            json.dump({"results": results, "total_cost_usd": round(total_cost, 4)}, f, indent=2)

    # Summary
    valid_results = [r for r in results if "cosine_similarity" in r]
    n_preserved   = sum(1 for r in valid_results if r["answer_preserved"])
    preservation_rate = n_preserved / max(1, len(valid_results))
    avg_reduction = sum(r["token_reduction_pct"] for r in valid_results) / max(1, len(valid_results))
    avg_similarity = sum(r["cosine_similarity"] for r in valid_results) / max(1, len(valid_results))

    failed_cases = [r for r in valid_results if not r["answer_preserved"]]

    summary = {
        "n_conversations": len(valid_results),
        "answer_preservation_rate": round(preservation_rate, 4),
        "avg_token_reduction_pct": round(avg_reduction, 4),
        "avg_cosine_similarity": round(avg_similarity, 4),
        "n_failed": len(failed_cases),
        "total_cost_usd": round(total_cost, 4),
    }

    with open(FUNCTIONAL_RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Layer 2 Results — Functional Validation")
    print(f"{'='*60}")
    print(f"  Answer preservation rate: {preservation_rate:.1%}  ({n_preserved}/{len(valid_results)})")
    print(f"  Average context reduction: {avg_reduction:.1%}")
    print(f"  Average cosine similarity: {avg_similarity:.3f}")
    print(f"  Failed cases (sim < {SIMILARITY_THRESHOLD}): {len(failed_cases)}")

    if failed_cases:
        print(f"\n  ── Failed cases for inspection ──")
        for r in failed_cases:
            print(f"\n  Conv: {r['conversation_id']} | SCR: {r['scr']:.0%} | sim={r['cosine_similarity']:.3f}")
            print(f"    Full answer:       {r['answer_full'][:200]}")
            print(f"    Compressed answer: {r['answer_compressed'][:200]}")

    print(f"\n  Total API cost: ${total_cost:.4f}")
    print(f"  Results saved → {FUNCTIONAL_RESULTS_PATH}")

    return summary


# ---------------------------------------------------------------------------
# Layer 3: Human spot-check report
# ---------------------------------------------------------------------------

def run_human_review(n: int = 25, dry_run: bool = False) -> dict:
    """
    Generate a readable Markdown file with n randomly sampled sentence-level decisions.
    """
    print(f"\n{'='*60}")
    print(f"Layer 3: Human Review — {n} random sentence decisions")
    print(f"{'='*60}")

    if dry_run:
        print(f"\n💰 Layer 3 is free (no API calls)")
        return {"dry_run": True, "estimated_cost_usd": 0.0}

    annotations = _load_annotations()
    if not annotations:
        print("❌ No annotations found.")
        sys.exit(1)

    # Collect all sentence-level decisions across all conversations
    all_decisions: list[dict] = []
    for ann in annotations:
        conv_id   = ann["conversation_id"]
        scr       = ann.get("safe_compression_rate", 0.0)
        sentences = ann.get("sentences", [])
        for sent_ann in ann.get("annotations", []):
            sid   = sent_ann["sentence_id"]
            label = sent_ann["label"]
            reason = sent_ann.get("reason", "")
            if sid < len(sentences):
                all_decisions.append({
                    "conv_id": conv_id,
                    "scr": scr,
                    "sentence_id": sid,
                    "label": label,
                    "reason": reason,
                    "text": sentences[sid],
                })

    rng = random.Random(SEED + 99)
    rng.shuffle(all_decisions)
    sample = all_decisions[:n]

    lines = [
        "# Sentence Review — Human Spot-Check",
        f"\n*Generated from {len(annotations)} annotated conversations*",
        f"*{n} random sentence-level decisions sampled for review*",
        "\n**Instructions:** For each decision below, mark whether you agree with the label.",
        "After review, % agree = human validation accuracy.\n",
        "---\n",
    ]

    for i, d in enumerate(sample, 1):
        scr_pct = f"{d['scr']:.0%}"
        label   = d["label"].upper()
        reason  = d["reason"]
        text    = d["text"]
        sid     = d["sentence_id"]

        # Truncate very long sentences for readability
        if len(text) > 300:
            text = text[:297] + "..."

        lines.append(f"### [{i:02d}] Conv: `{d['conv_id']}` | SCR: {scr_pct}")
        lines.append(f"**[{sid}] {label}** — \"{reason}\"")
        lines.append(f"> {text}")
        lines.append(f"\nAgree? [ ] Yes &nbsp;&nbsp; [ ] No — if no, correct label: `_______________`\n")

    # Summary section at end
    lines.append("---\n")
    lines.append("## Summary\n")
    lines.append("After completing review above, fill in:\n")
    lines.append("- **Total reviewed:** _____ / " + str(n))
    lines.append("- **Agreed with GPT-5.2:** _____")
    lines.append("- **Human validation accuracy:** _____ %")

    HUMAN_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")

    label_counts = {}
    for d in sample:
        label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1

    print(f"  Generated {n} sentence decisions")
    print(f"  Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")
    print(f"  Saved → {HUMAN_REVIEW_PATH}")

    return {
        "n_samples": n,
        "label_counts": label_counts,
        "output_path": str(HUMAN_REVIEW_PATH),
    }


# ---------------------------------------------------------------------------
# Cost estimate for dry-run
# ---------------------------------------------------------------------------

def print_cost_estimate(iaa_n: int = 30, func_n: int = 20) -> None:
    print(f"\n{'='*60}")
    print(f"  💰 Cost Estimate — Full Verification Suite")
    print(f"{'='*60}")

    # Layer 1
    l1_in   = iaa_n * 800
    l1_out  = iaa_n * 600
    l1_cost = (l1_in / 1e6) * CLAUDE_INPUT_PRICE + (l1_out / 1e6) * CLAUDE_OUTPUT_PRICE

    # Layer 2
    l2_in   = func_n * 2 * 500
    l2_out  = func_n * 2 * 300
    l2_cost = (l2_in / 1e6) * CLAUDE_INPUT_PRICE + (l2_out / 1e6) * CLAUDE_OUTPUT_PRICE

    total   = l1_cost + l2_cost

    print(f"\n  Layer 1 (IAA, {iaa_n} conversations):")
    print(f"    Input:  {iaa_n} × 800 = {l1_in:,} tokens → ${l1_in/1e6*CLAUDE_INPUT_PRICE:.4f}")
    print(f"    Output: {iaa_n} × 600 = {l1_out:,} tokens → ${l1_out/1e6*CLAUDE_OUTPUT_PRICE:.4f}")
    print(f"    Subtotal: ~${l1_cost:.4f}")

    print(f"\n  Layer 2 (Functional, {func_n} conversations × 2 calls):")
    print(f"    Input:  {func_n}×2 × 500 = {l2_in:,} tokens → ${l2_in/1e6*CLAUDE_INPUT_PRICE:.4f}")
    print(f"    Output: {func_n}×2 × 300 = {l2_out:,} tokens → ${l2_out/1e6*CLAUDE_OUTPUT_PRICE:.4f}")
    print(f"    Subtotal: ~${l2_cost:.4f}")

    print(f"\n  Layer 3 (Human review): FREE (no API calls)")
    print(f"\n  ── Total estimated cost: ~${total:.4f} ──")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CompressionSafety Verification Suite — IAA, functional, and human review layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # iaa
    p_iaa = sub.add_parser("iaa", help="Layer 1: IAA with Claude Sonnet 4.6")
    p_iaa.add_argument("--n", type=int, default=30, help="Number of conversations to sample")
    p_iaa.add_argument("--dry-run", action="store_true")

    # functional
    p_func = sub.add_parser("functional", help="Layer 2: Compression-then-test")
    p_func.add_argument("--n", type=int, default=20, help="Number of conversations")
    p_func.add_argument("--dry-run", action="store_true")

    # human-review
    p_review = sub.add_parser("human-review", help="Layer 3: Generate human review Markdown")
    p_review.add_argument("--n", type=int, default=25, help="Number of sentence samples")
    p_review.add_argument("--dry-run", action="store_true")

    # all
    p_all = sub.add_parser("all", help="Run all three layers")
    p_all.add_argument("--iaa-n",    type=int, default=30)
    p_all.add_argument("--func-n",   type=int, default=20)
    p_all.add_argument("--review-n", type=int, default=25)
    p_all.add_argument("--dry-run",  action="store_true")

    args = parser.parse_args()

    # Ensure output dir exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "iaa":
        run_iaa(n=args.n, dry_run=args.dry_run)

    elif args.command == "functional":
        run_functional(n=args.n, dry_run=args.dry_run)

    elif args.command == "human-review":
        run_human_review(n=args.n, dry_run=args.dry_run)

    elif args.command == "all":
        if args.dry_run:
            print_cost_estimate(iaa_n=args.iaa_n, func_n=args.func_n)
            run_iaa(n=args.iaa_n, dry_run=True)
            run_functional(n=args.func_n, dry_run=True)
            run_human_review(n=args.review_n, dry_run=True)
        else:
            api_key = _load_openrouter_key()
            run_iaa(n=args.iaa_n, api_key=api_key)
            run_functional(n=args.func_n, api_key=api_key)
            run_human_review(n=args.review_n)
            print(f"\n✅ All layers complete!")


if __name__ == "__main__":
    main()
