"""
Experiment 4: Agentic FRAMES Benchmark — ContextPrune Compression & Accuracy

Tests 4 compression conditions on the frames_agentic dataset where each question is wrapped
in a realistic multi-layered agentic context (system + memory_block + tool_output + passage),
creating ~3-4x redundancy specifically designed to test semantic deduplication.

Conditions:
  1. raw         — full redundant context (system + memory_block + tool_output + passage + question)
  2. truncation  — truncate to 40% of tokens
  3. llmlingua2  — use cached LLMLingua-2 compressed version (skip on cache miss)
  4. contextprune — run ContextPrune semantic dedup

Models: claude-sonnet-4-6, gemini-3.1-pro-preview, grok-4.1-fast, gpt-5.2
         (NO kimi, NO codex)

Usage:
  # Quick 3-question validation (default --n 10)
  python3 benchmarks/exp4_agentic.py --n 3

  # Specify models
  python3 benchmarks/exp4_agentic.py --n 3 --models claude,gpt52

  # Full run
  python3 benchmarks/exp4_agentic.py --n 100 --models all --budget 25.00

  # Dry run (no API calls)
  python3 benchmarks/exp4_agentic.py --n 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextprune.adapters.openrouter import OpenRouterAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

AGENTIC_DATA_PATH = DATA_DIR / "frames_agentic" / "test.jsonl"

CONDITIONS = ["raw", "truncation", "llmlingua2", "contextprune"]

# Models for exp4 (NO kimi, NO codex per spec)
EXP4_MODELS: dict[str, str] = {
    "claude":  "anthropic/claude-sonnet-4-6",
    "gemini":  "google/gemini-3.1-pro-preview",
    "grok":    "x-ai/grok-4.1-fast",
    "gpt52":   "openai/gpt-5.2",
}

PRICING: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4-6":   {"input":  3.00, "output": 15.00},
    "google/gemini-3.1-pro-preview":  {"input":  2.00, "output": 12.00},
    "x-ai/grok-4.1-fast":            {"input":  0.20, "output":  0.50},
    "openai/gpt-5.2":                {"input":  1.75, "output": 14.00},
}


# ---------------------------------------------------------------------------
# Compression helpers (copied from exp3_accuracy.py)
# ---------------------------------------------------------------------------

def apply_truncation(messages: list[dict], system: str, target_tokens: int) -> tuple[list[dict], str]:
    """Truncate from the END to match target_tokens. Preserves final user message (question)."""
    def tok(s: str) -> int:
        return max(1, len(s) // 4)

    question = messages[-1] if messages else {"role": "user", "content": ""}
    non_question = messages[:-1]
    q_tokens = tok(question.get("content", ""))
    sys_tokens = tok(system)
    remaining = target_tokens - q_tokens - sys_tokens

    if remaining <= 0:
        return [question], system

    kept = []
    used = 0
    for msg in reversed(non_question):
        content = msg.get("content", "")
        t = tok(content)
        if used + t <= remaining:
            kept.insert(0, msg)
            used += t
        else:
            chars_left = (remaining - used) * 4
            if chars_left > 50:
                trimmed = dict(msg)
                trimmed["content"] = content[-chars_left:]
                kept.insert(0, trimmed)
            break

    return kept + [question], system


LLMLINGUA2_CACHE_MISS_SENTINEL = "__cache_miss__"


def apply_llmlingua2(
    messages: list[dict],
    system: str,
    compression_ratio: float = 0.5,
    item_id: str | None = None,
    llmlingua2_cache: dict | None = None,
) -> tuple[list[dict], str] | str:
    """
    Apply LLMLingua-2 compression from pre-computed cache only.
    Returns LLMLINGUA2_CACHE_MISS_SENTINEL string on cache miss (caller skips this item/condition).
    Per spec: 'use cached LLMLingua-2 compressed version (or skip if cache miss)'
    """
    # Cache lookup only — no live inference for exp4
    if llmlingua2_cache and item_id:
        cached = llmlingua2_cache.get(item_id)
        if cached:
            cached_msgs = cached.get("compressed_messages", messages)
            if cached_msgs and messages:
                cached_msgs = list(cached_msgs[:-1]) + [messages[-1]]
            return (cached_msgs, cached.get("compressed_system", system))

    # Cache miss → signal caller to skip
    return LLMLINGUA2_CACHE_MISS_SENTINEL


def apply_contextprune(messages: list[dict], system: str) -> tuple[list[dict], str, dict]:
    """Apply ContextPrune semantic dedup + MMR compression pipeline."""
    from contextprune import Config
    from contextprune.dedup import SemanticDeduplicator, MMRSelector
    from contextprune.tokenizer import count_message_tokens, count_system_tokens

    config = Config(semantic_dedup=True, tool_filter=True, budget_injection=False, use_mmr=True)
    dedup = SemanticDeduplicator(
        similarity_threshold=config.similarity_threshold,
        model=config.dedup_model,
    )
    mmr = MMRSelector(
        token_budget_ratio=config.mmr_token_budget_ratio,
        lambda_param=config.mmr_lambda,
        model=config.dedup_model,
        min_tokens_to_mmr=config.mmr_min_tokens,
    )

    orig_tokens = count_message_tokens(messages) + count_system_tokens(system)

    # Step 1: MMR on system
    new_system = system
    if system and isinstance(system, str):
        sys_tok = count_system_tokens(system)
        if sys_tok > config.mmr_min_tokens:
            from contextprune.core import _get_query_signal
            query = _get_query_signal(messages)
            compressed_sys, _ = mmr.select(system, query)
            new_system = compressed_sys

    # Step 2: Semantic dedup
    new_messages, new_system_dedup, removed = dedup.deduplicate(messages, system=new_system if isinstance(new_system, str) else None)
    if new_system_dedup and isinstance(new_system, str):
        new_system = new_system_dedup

    # Step 3: Per-message MMR
    from contextprune.core import _get_query_signal
    query = _get_query_signal(new_messages)
    final_messages = []
    for msg in new_messages:
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            msg_tok = count_message_tokens([msg])
            if msg_tok > config.mmr_min_tokens:
                compressed, _ = mmr.select(content, query)
                msg = dict(msg)
                msg["content"] = compressed
        final_messages.append(msg)

    comp_tokens = count_message_tokens(final_messages) + count_system_tokens(new_system)
    stats = {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "savings_pct": round(max(0, (orig_tokens - comp_tokens) / max(1, orig_tokens) * 100), 1),
        "dedup_removed": removed,
    }
    return final_messages, new_system, stats


# ---------------------------------------------------------------------------
# Accuracy evaluator (substring match for agentic FRAMES)
# ---------------------------------------------------------------------------

def eval_frames_agentic(response: str, gold_answer: str) -> bool:
    """
    Evaluate accuracy: check if the model's response contains key words from the gold answer.
    Case-insensitive substring match. Also checks individual significant tokens.
    """
    response_lower = response.lower().strip()
    gold_lower = gold_answer.lower().strip()

    # Direct substring match
    if gold_lower in response_lower:
        return True

    # Clean gold answer (remove punctuation for token match)
    gold_clean = re.sub(r"[^\w\s]", " ", gold_lower).strip()
    response_clean = re.sub(r"[^\w\s]", " ", response_lower).strip()

    if gold_clean in response_clean:
        return True

    # Token overlap: check if significant gold tokens appear in response
    # "Significant" = length > 3 (ignore stop words like "the", "is", "a")
    gold_tokens = [t for t in gold_clean.split() if len(t) > 3]
    if not gold_tokens:
        # Short answer: exact match
        return gold_clean.strip() == response_clean.strip()

    # All significant tokens must appear in response
    return all(tok in response_clean for tok in gold_tokens)


# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------

def load_openrouter_key() -> str | None:
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
# Dataset loader
# ---------------------------------------------------------------------------

def load_agentic_dataset(n: int) -> list[dict]:
    """Load n items from frames_agentic test.jsonl."""
    if not AGENTIC_DATA_PATH.exists():
        print(f"ERROR: {AGENTIC_DATA_PATH} not found.", file=sys.stderr)
        print("Run: python3 benchmarks/create_agentic_dataset.py", file=sys.stderr)
        sys.exit(1)

    items: list[dict] = []
    with open(AGENTIC_DATA_PATH, encoding="utf-8") as f:
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

    return items


# ---------------------------------------------------------------------------
# Build messages from agentic context
# ---------------------------------------------------------------------------

def build_messages_from_item(item: dict) -> tuple[list[dict], str]:
    """
    Build (messages, system) from an agentic context item.
    Layout:
      system = agentic_context['system']
      messages = [
        user: memory_block,
        assistant: tool_output,
        user: retrieval_passage,
        user: question
      ]
    """
    ctx = item["agentic_context"]
    system = ctx["system"]
    messages = [
        {"role": "user", "content": ctx["memory_block"]},
        {"role": "assistant", "content": ctx["tool_output"]},
        {"role": "user", "content": ctx["retrieval_passage"]},
        {"role": "user", "content": item["question"]},
    ]
    return messages, system


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class Checkpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._completed: set[tuple[str, str, str]] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    self._completed.add((r["question_id"], r["condition"], r["model"]))
                except Exception:
                    pass
        if self._completed:
            print(f"  [checkpoint] Resumed — {len(self._completed):,} results already recorded.")

    def is_done(self, question_id: str, condition: str, model: str) -> bool:
        return (question_id, condition, model) in self._completed

    def write(self, result: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        self._completed.add((result["question_id"], result["condition"], result["model"]))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n: int,
    model_aliases: list[str],
    conditions: list[str],
    budget_usd: float,
    dry_run: bool = False,
    skip_llmlingua2: bool = False,
) -> list[dict]:
    """Run exp4 agentic benchmark."""

    if skip_llmlingua2:
        conditions = [c for c in conditions if c != "llmlingua2"]
        print("  [skip-llmlingua2] LLMLingua-2 condition skipped.")

    # Resolve model IDs
    model_ids: list[str] = []
    for alias in model_aliases:
        if alias == "all":
            model_ids.extend(EXP4_MODELS.values())
        elif alias in EXP4_MODELS:
            model_ids.append(EXP4_MODELS[alias])
        elif "/" in alias:
            model_ids.append(alias)
        else:
            print(f"  [WARN] Unknown model alias: {alias!r} — skipping")
    # Dedupe while preserving order
    seen: set[str] = set()
    model_ids = [m for m in model_ids if not (m in seen or seen.add(m))]

    if not model_ids:
        print("ERROR: No valid models specified.", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    print(f"\nLoading {n} items from frames_agentic…")
    items = load_agentic_dataset(n)
    print(f"  Loaded {len(items)} items")

    # Load LLMLingua-2 cache (empty — no pre-computed cache for frames_agentic yet)
    llmlingua2_cache: dict = {}
    cache_path = DATA_DIR / "frames_agentic" / "llmlingua2_compressed.jsonl"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "id" in rec and "error" not in rec:
                        llmlingua2_cache[rec["id"]] = rec
                except Exception:
                    pass
        if llmlingua2_cache:
            print(f"  [llmlingua2] Loaded {len(llmlingua2_cache):,} cached items.")

    # Checkpoint
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = RESULTS_DIR / f"exp4_agentic_{ts}.jsonl"
    ckpt = Checkpoint(ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # API adapter
    adapter: OpenRouterAdapter | None = None
    if not dry_run:
        api_key = load_openrouter_key()
        if not api_key:
            print("ERROR: No OPENROUTER_API_KEY found. Use --dry-run or set env var.", file=sys.stderr)
            sys.exit(1)
        adapter = OpenRouterAdapter(api_key=api_key)

    spent = 0.0
    results: list[dict] = []

    total_calls = len(items) * len(conditions) * len(model_ids)
    call_num = 0

    print(f"\n{'DRY RUN — ' if dry_run else ''}Running {len(items)} questions × {len(conditions)} conditions × {len(model_ids)} models = {total_calls} calls\n")

    for item_idx, item in enumerate(items):
        qid = item.get("id", f"q_{item_idx:05d}")
        messages_raw, system_raw = build_messages_from_item(item)
        gold_answer = item.get("answer", "")

        for condition in conditions:
            for model_id in model_ids:
                model_alias = next((k for k, v in EXP4_MODELS.items() if v == model_id), model_id.split("/")[-1])
                call_num += 1

                if ckpt.is_done(qid, condition, model_alias):
                    continue

                # Apply compression for condition
                comp_messages = messages_raw
                comp_system = system_raw
                comp_stats: dict = {}
                comp_ratio = 1.0

                t_compress = time.perf_counter()
                orig_tok = sum(len(m.get("content", "")) // 4 for m in messages_raw) + len(system_raw) // 4

                if condition == "raw":
                    comp_messages, comp_system = messages_raw, system_raw
                    comp_ratio = 1.0

                elif condition == "truncation":
                    target = max(100, int(orig_tok * 0.40))
                    comp_messages, comp_system = apply_truncation(messages_raw, system_raw, target)
                    comp_tok = sum(len(m.get("content", "")) // 4 for m in comp_messages) + len(comp_system) // 4
                    comp_ratio = comp_tok / max(1, orig_tok)

                elif condition == "llmlingua2":
                    ll2_result = apply_llmlingua2(
                        messages_raw, system_raw,
                        compression_ratio=0.45,
                        item_id=qid,
                        llmlingua2_cache=llmlingua2_cache,
                    )
                    if ll2_result == LLMLINGUA2_CACHE_MISS_SENTINEL:
                        # Skip this item for llmlingua2 (no pre-computed cache)
                        print(f"  Q{item_idx+1:03d} [{condition:12s}] [{model_alias:7s}] "
                              f"SKIP (no cache)")
                        continue
                    comp_messages, comp_system = ll2_result
                    comp_tok = sum(len(m.get("content", "")) // 4 for m in comp_messages) + len(comp_system) // 4
                    comp_ratio = comp_tok / max(1, orig_tok)

                elif condition == "contextprune":
                    try:
                        comp_messages, comp_system, cp_stats = apply_contextprune(messages_raw, system_raw)
                        comp_stats = cp_stats
                        orig_cp = cp_stats.get("original_tokens", orig_tok)
                        comp_cp = cp_stats.get("compressed_tokens", orig_cp)
                        comp_ratio = round(comp_cp / max(1, orig_cp), 4)
                    except Exception as e:
                        print(f"  [WARN] ContextPrune failed on {qid}: {e}")
                        comp_messages, comp_system = messages_raw, system_raw
                        comp_ratio = 1.0

                compress_ms = (time.perf_counter() - t_compress) * 1000
                tokens_in_est = sum(len(m.get("content", "")) // 4 for m in comp_messages) + len(comp_system) // 4

                # Cost estimate
                pricing = PRICING.get(model_id, {"input": 2.0, "output": 10.0})
                est_cost = (tokens_in_est / 1e6) * pricing["input"] + (256 / 1e6) * pricing["output"]

                if dry_run:
                    result = {
                        "question_id": qid,
                        "condition": condition,
                        "model": model_alias,
                        "correct": None,
                        "tokens_in": tokens_in_est,
                        "tokens_out": 0,
                        "cost_usd": 0.0,
                        "latency_ms": 0.0,
                        "compress_ms": round(compress_ms, 2),
                        "compression_ratio": round(comp_ratio, 4),
                        "dry_run": True,
                        "question": item["question"][:80],
                        "answer": gold_answer[:60],
                    }
                    results.append(result)
                    print(f"  Q{item_idx+1:03d} [{condition:12s}] [{model_alias:7s}] "
                          f"DRY {tokens_in_est:,}tok ratio={comp_ratio:.2f}")
                    continue

                # Budget check
                if spent + est_cost > budget_usd:
                    print(f"\n  [BUDGET] ${budget_usd:.2f} limit reached — stopping.")
                    return results

                # API call
                t_api = time.perf_counter()
                try:
                    completion = adapter.complete(
                        messages=comp_messages,
                        model=model_id,
                        system=comp_system,
                        max_tokens=512,
                        temperature=0.0,
                    )
                    latency_ms = (time.perf_counter() - t_api) * 1000
                    response_text = completion.text
                    actual_cost = completion.cost_usd
                    tokens_out = completion.output_tokens
                    tokens_in_actual = completion.input_tokens
                    spent += actual_cost

                    correct = eval_frames_agentic(response_text, gold_answer)

                    result = {
                        "question_id": qid,
                        "condition": condition,
                        "model": model_alias,
                        "correct": bool(correct),
                        "tokens_in": tokens_in_actual,
                        "tokens_out": tokens_out,
                        "cost_usd": actual_cost,
                        "latency_ms": round(latency_ms, 2),
                        "compress_ms": round(compress_ms, 2),
                        "compression_ratio": round(comp_ratio, 4),
                        "question": item["question"][:120],
                        "answer": gold_answer[:80],
                        "response_preview": response_text[:200],
                    }
                    ckpt.write(result)
                    results.append(result)

                    correct_str = "✓" if correct else "✗"
                    print(f"  Q{item_idx+1:03d} [{condition:12s}] [{model_alias:7s}] "
                          f"{correct_str} {tokens_in_actual:,}tok ${actual_cost:.5f} "
                          f"{latency_ms:.0f}ms ratio={comp_ratio:.2f} spent=${spent:.4f}")

                except Exception as exc:
                    latency_ms = (time.perf_counter() - t_api) * 1000
                    print(f"  [ERROR] {qid}/{condition}/{model_alias}: {exc}")
                    result = {
                        "question_id": qid,
                        "condition": condition,
                        "model": model_alias,
                        "correct": False,
                        "tokens_in": tokens_in_est,
                        "tokens_out": 0,
                        "cost_usd": 0.0,
                        "latency_ms": round(latency_ms, 2),
                        "compress_ms": round(compress_ms, 2),
                        "compression_ratio": round(comp_ratio, 4),
                        "error": str(exc)[:200],
                        "question": item["question"][:120],
                        "answer": gold_answer[:80],
                    }
                    ckpt.write(result)
                    results.append(result)

    return results


# ---------------------------------------------------------------------------
# Summary / stats
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], dry_run: bool = False) -> None:
    """Print accuracy and compression stats per condition."""
    if not results:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 65)
    print("Experiment 4 — Agentic FRAMES Summary")
    print("=" * 65)

    if dry_run:
        n_q = len(set(r["question_id"] for r in results))
        conditions = sorted(set(r["condition"] for r in results))
        models = sorted(set(r["model"] for r in results))
        print(f"  DRY RUN: {n_q} questions × {len(conditions)} conditions × {len(models)} models")
        print(f"  Total calls that would be made: {len(results)}")
        # Compression ratios per condition
        print("\n  Compression ratio by condition (estimated):")
        for cond in conditions:
            cond_results = [r for r in results if r["condition"] == cond]
            if cond_results:
                ratios = [r["compression_ratio"] for r in cond_results]
                avg = sum(ratios) / len(ratios)
                print(f"    {cond:15s}: avg ratio = {avg:.3f}")
        return

    # --- Accuracy per (condition, model) ---
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    ratios_by_cond: dict[str, list[float]] = defaultdict(list)

    for r in results:
        if r.get("dry_run"):
            continue
        key = (r["condition"], r["model"])
        correct = r.get("correct")
        if correct is not None:
            groups[key].append(bool(correct))
        if "compression_ratio" in r:
            ratios_by_cond[r["condition"]].append(r["compression_ratio"])

    conditions_seen = sorted({k[0] for k in groups})
    models_seen = sorted({k[1] for k in groups})

    # Accuracy table
    print("\n  Accuracy by condition × model:")
    header = f"  {'Condition':<15}" + "".join(f"  {m:>10}" for m in models_seen)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cond in conditions_seen:
        row = f"  {cond:<15}"
        for model in models_seen:
            key = (cond, model)
            if key in groups and groups[key]:
                acc = sum(groups[key]) / len(groups[key])
                row += f"  {acc:>9.1%}"
            else:
                row += f"  {'N/A':>9}"
        print(row)

    # Compression ratios
    print("\n  Compression ratio by condition:")
    for cond in conditions_seen:
        rs = ratios_by_cond.get(cond, [])
        if rs:
            avg = sum(rs) / len(rs)
            mn = min(rs)
            mx = max(rs)
            contextprune_flag = " ← target < 0.70" if cond == "contextprune" else ""
            print(f"    {cond:<15}: avg={avg:.3f}  min={mn:.3f}  max={mx:.3f}{contextprune_flag}")

    # Cost summary
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    print(f"\n  Total cost: ${total_cost:.4f}")

    # Overall stats
    print("\n  Per-condition summary:")
    for cond in conditions_seen:
        cond_results = [r for r in results if r["condition"] == cond and not r.get("dry_run")]
        if not cond_results:
            continue
        correct_list = [r["correct"] for r in cond_results if r.get("correct") is not None]
        acc = sum(bool(c) for c in correct_list) / max(1, len(correct_list))
        rs = ratios_by_cond.get(cond, [])
        avg_ratio = sum(rs) / max(1, len(rs))
        print(f"    {cond:<15}: accuracy={acc:.1%}  n={len(correct_list)}  avg_ratio={avg_ratio:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp 4: Agentic FRAMES compression + accuracy benchmark.")
    p.add_argument("--n", type=int, default=10, help="Number of questions to run (default: 10)")
    p.add_argument("--models", default="claude,gemini,grok,gpt52",
                   help="Comma-separated model aliases or 'all'. Choices: " + ", ".join(EXP4_MODELS))
    p.add_argument("--conditions", default=",".join(CONDITIONS),
                   help="Conditions to run (default: all 4)")
    p.add_argument("--budget", type=float, default=10.00, help="Budget in USD (default $10.00)")
    p.add_argument("--dry-run", action="store_true", help="Validate pipeline without API calls")
    p.add_argument("--skip-llmlingua2", action="store_true",
                   help="Skip LLMLingua-2 condition (useful for fast validation)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (unused — deterministic order)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_aliases = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    results = run_experiment(
        n=args.n,
        model_aliases=model_aliases,
        conditions=conditions,
        budget_usd=args.budget,
        dry_run=args.dry_run,
        skip_llmlingua2=args.skip_llmlingua2,
    )

    print_summary(results, dry_run=args.dry_run)

    if args.dry_run:
        print("\n✓ Dry run complete — pipeline validated (no API calls).")
    else:
        print(f"\n✓ Experiment complete. {len(results)} results recorded.")


if __name__ == "__main__":
    main()
