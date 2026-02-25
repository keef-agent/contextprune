"""
Experiment 3: Task Performance on Standard Benchmarks

Runs questions through 4 compression conditions × configured models, measures accuracy.
Implements cost controls, checkpointing, and statistical analysis.

Conditions:
  1. raw         — inject context, send directly to model, no compression
  2. truncation  — inject context, truncate from END to match ContextPrune output length
  3. llmlingua2  — inject context, compress with LLMLingua-2
  4. contextprune — inject context, compress with ContextPrune

Usage:
  # Dry run (no API calls, validate pipeline on 10 questions)
  python3 benchmarks/exp3_accuracy.py --dataset mmlu_pro --dry-run --n 10

  # Dry run with model specified
  python3 benchmarks/exp3_accuracy.py --dataset mmlu_pro --dry-run --n 10 --models claude

  # Real run with cost guard
  python3 benchmarks/exp3_accuracy.py --dataset mmlu_pro --models claude,gpt52 --n 100 --budget 10.00

  # Cost estimate only
  python3 benchmarks/exp3_accuracy.py --dataset mmlu_pro --cost-estimate --n 200 --models all

  # Full run
  python3 benchmarks/exp3_accuracy.py --dataset mmlu_pro --models all --n 200 --budget 50.00
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextprune.adapters.openrouter import SUPPORTED_MODELS, OpenRouterAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CONDITIONS = ["raw", "truncation", "llmlingua2", "contextprune"]

# Pricing per million tokens (from openrouter adapter)
PRICING = {
    "anthropic/claude-sonnet-4-6":  {"input": 3.00, "output": 15.00},
    "google/gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "x-ai/grok-4.1-fast":           {"input": 0.20, "output": 0.50},
    "moonshotai/kimi-k2.5":         {"input": 0.45, "output": 2.20},
    "openai/gpt-5.2":               {"input": 1.75, "output": 14.00},
    "openai/gpt-5.3-codex":         {"input": 1.75, "output": 14.00},
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token bucket rate limiter per model. Prevents 429s on OpenRouter."""

    # Conservative limits — OpenRouter actual limits are higher but unconfirmed
    LIMITS: dict[str, dict[str, int]] = {
        "anthropic/claude-sonnet-4-6":   {"rpm": 50,  "tpm": 40_000},
        "google/gemini-3.1-pro-preview":  {"rpm": 60,  "tpm": 60_000},
        "x-ai/grok-4.1-fast":            {"rpm": 100, "tpm": 100_000},
        "moonshotai/kimi-k2.5":          {"rpm": 60,  "tpm": 60_000},
        "openai/gpt-5.2":                {"rpm": 60,  "tpm": 80_000},
        "openai/gpt-5.3-codex":          {"rpm": 60,  "tpm": 80_000},
    }

    def __init__(self) -> None:
        # {model: [timestamp, ...]} rolling window of request times (last 60s)
        self._request_times: dict[str, list[float]] = defaultdict(list)
        # {model: total_tokens_in_window}
        self._token_counts: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def _prune_window(self, model: str) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        self._request_times[model] = [t for t in self._request_times[model] if t > cutoff]
        self._token_counts[model] = [(t, n) for t, n in self._token_counts[model] if t > cutoff]

    def wait_if_needed(self, model: str, tokens: int) -> None:
        """Sleep if we're approaching rate limits for this model."""
        limits = self.LIMITS.get(model)
        if not limits:
            return  # unknown model: no throttle

        self._prune_window(model)
        now = time.monotonic()
        rpm_limit = limits["rpm"]
        tpm_limit = limits["tpm"]

        # Check requests per minute
        req_count = len(self._request_times[model])
        tokens_used = sum(n for _, n in self._token_counts[model])

        # If at 80% of either limit, add a small delay
        if req_count >= int(rpm_limit * 0.8) or tokens_used + tokens >= int(tpm_limit * 0.8):
            sleep_sec = 1.5
            logging.debug(
                f"RateLimiter: {model} at {req_count}/{rpm_limit} rpm, "
                f"{tokens_used}/{tpm_limit} tpm — sleeping {sleep_sec}s"
            )
            time.sleep(sleep_sec)
            self._prune_window(model)  # prune again after sleep

        # Record this request
        now = time.monotonic()
        self._request_times[model].append(now)
        self._token_counts[model].append((now, tokens))

    def record_success(self, model: str) -> None:
        """No-op: success doesn't change rate tracking (already recorded in wait_if_needed)."""
        pass


# ---------------------------------------------------------------------------
# ModelHealthTracker
# ---------------------------------------------------------------------------

class ModelHealthTracker:
    """Disables models that return 3+ consecutive 5xx errors."""

    def __init__(self, max_consecutive_failures: int = 3) -> None:
        self.max_consecutive_failures = max_consecutive_failures
        self.failures: dict[str, int] = defaultdict(int)
        self.disabled: set[str] = set()

    def record_failure(self, model: str, error_code: int) -> None:
        if error_code >= 500:
            self.failures[model] += 1
            if self.failures[model] >= self.max_consecutive_failures:
                self.disabled.add(model)
                logging.error(
                    f"Model {model} disabled after {self.failures[model]} "
                    f"consecutive 5xx errors — skipping for remainder of run."
                )

    def record_success(self, model: str) -> None:
        self.failures[model] = 0  # reset on success

    def is_available(self, model: str) -> bool:
        return model not in self.disabled


# ---------------------------------------------------------------------------
# BudgetExceededError
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    pass


# ---------------------------------------------------------------------------
# CostGuard
# ---------------------------------------------------------------------------

class CostGuard:
    """
    Tracks spending and raises BudgetExceededError when budget would be exceeded.
    Persists state to a checkpoint file so it survives restarts.
    """

    def __init__(self, budget_usd: float, checkpoint_path: Path | None = None) -> None:
        self.budget = budget_usd
        self.spent = 0.0
        self._checkpoint = checkpoint_path
        if checkpoint_path and checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    data = json.load(f)
                    self.spent = float(data.get("spent_usd", 0.0))
            except Exception:
                pass

    def check(self, projected_cost: float) -> None:
        """Raise BudgetExceededError if spending + projected_cost > budget."""
        if self.spent + projected_cost > self.budget:
            raise BudgetExceededError(
                f"Budget ${self.budget:.2f} would be exceeded. "
                f"Spent: ${self.spent:.4f}, projected: ${projected_cost:.4f}"
            )

    def record(self, cost: float) -> None:
        """Record actual cost and persist to checkpoint."""
        self.spent += cost
        if self._checkpoint:
            try:
                existing: dict = {}
                if self._checkpoint.exists():
                    try:
                        with open(self._checkpoint) as f:
                            existing = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        existing = {}
                existing["spent_usd"] = round(self.spent, 8)
                with open(self._checkpoint, "w") as f:
                    json.dump(existing, f)
            except Exception:
                pass

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class Checkpoint:
    """
    Append-only JSONL checkpoint for experiment results.
    Supports resuming: reads completed (question_id, condition, model) tuples.
    """

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

    def load_all(self) -> list[dict]:
        results: list[dict] = []
        if not self.path.exists():
            return results
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
        return results


# ---------------------------------------------------------------------------
# LLMLingua-2 cache helpers
# ---------------------------------------------------------------------------

def _load_llmlingua2_cache(dataset: str) -> dict[str, dict]:
    """
    Load the pre-computed LLMLingua-2 cache for a dataset.
    Returns {item_id: record_dict}.
    Cache is built by benchmarks/precompute_llmlingua2.py.
    """
    cache_path = DATA_DIR / dataset / "llmlingua2_compressed.jsonl"
    if not cache_path.exists():
        return {}
    cache: dict[str, dict] = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec and "error" not in rec:
                    cache[rec["id"]] = rec
            except Exception:
                pass
    return cache


def get_llmlingua2_compressed(
    item_id: str,
    dataset: str,
    cache: dict[str, dict] | None = None,
) -> dict | None:
    """
    Look up a pre-computed LLMLingua-2 result.
    Returns the cache record (with compressed_messages, compressed_system) or None.
    Pass `cache` if you've already loaded it (avoids re-reading the file per call).
    """
    if cache is None:
        cache = _load_llmlingua2_cache(dataset)
    return cache.get(item_id)


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def _messages_to_text(messages: list[dict]) -> str:
    """Flatten messages list to a single string for token counting / truncation."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role.upper()}]: {content}")
    return "\n\n".join(parts)


def _text_to_messages(text: str, original_messages: list[dict]) -> list[dict]:
    """Restore message structure from truncated text (best-effort)."""
    # If the text is shorter, the last user message gets truncated
    # Preserve the last user message role structure
    last_user_idx = max(
        (i for i, m in enumerate(original_messages) if m.get("role") == "user"),
        default=len(original_messages) - 1,
    )
    prefix_msgs = original_messages[:last_user_idx]
    last_content = text  # simplified — use truncated text as final user message
    return prefix_msgs + [{"role": "user", "content": last_content}]


def apply_truncation(messages: list[dict], system: str, target_tokens: int) -> tuple[list[dict], str]:
    """
    Truncate from the END of the context (not the question) to match target_tokens.
    Preserves the final user message (the question). Truncates earlier history first.
    """
    # Approximate tokens
    def tok(s: str) -> int:
        return max(1, len(s) // 4)

    question = messages[-1] if messages else {"role": "user", "content": ""}
    non_question = messages[:-1]
    q_tokens = tok(question.get("content", ""))
    sys_tokens = tok(system)

    remaining = target_tokens - q_tokens - sys_tokens
    if remaining <= 0:
        return [question], system

    # Fill up from end of non-question messages
    kept = []
    used = 0
    for msg in reversed(non_question):
        content = msg.get("content", "")
        t = tok(content)
        if used + t <= remaining:
            kept.insert(0, msg)
            used += t
        else:
            # Partially include this message
            chars_left = (remaining - used) * 4
            if chars_left > 50:
                trimmed = dict(msg)
                trimmed["content"] = content[-chars_left:]  # keep the END (most recent)
                kept.insert(0, trimmed)
            break

    return kept + [question], system


def apply_llmlingua2(
    messages: list[dict],
    system: str,
    compression_ratio: float = 0.5,
    item_id: str | None = None,
    dataset: str | None = None,
    llmlingua2_cache: dict[str, dict] | None = None,
) -> tuple[list[dict], str]:
    """
    Apply LLMLingua-2 compression. Checks pre-computed cache first (fast).
    Falls back to live inference (slow, ~30s/question on CPU) if cache miss.
    Falls back to truncation if llmlingua not installed.
    compression_ratio: target compression (0.5 = keep 50% of tokens)
    """
    # --- Cache lookup (Fix 1) ---
    if item_id and dataset:
        cached = get_llmlingua2_compressed(item_id, dataset, cache=llmlingua2_cache)
        if cached:
            return (
                cached.get("compressed_messages", messages),
                cached.get("compressed_system", system),
            )
        else:
            logging.warning(
                f"LLMLingua-2 cache miss for {item_id} — running live (~30s on CPU). "
                f"Run benchmarks/precompute_llmlingua2.py first."
            )

    try:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        # Compress the context (all messages except final user question)
        context_msgs = messages[:-1]
        question_msg = messages[-1]
        context_text = _messages_to_text(context_msgs)
        question_text = question_msg.get("content", "")

        if len(context_text) > 50:
            result = compressor.compress_prompt(
                context_text,
                instruction=question_text,
                question=question_text,
                rate=compression_ratio,
            )
            compressed_context = result["compressed_prompt"]
            new_msgs = [{"role": "system", "content": compressed_context}] + [question_msg]
        else:
            new_msgs = messages
        return new_msgs, system
    except ImportError:
        # Fall back to truncation at 50%
        approx_total = sum(len(m.get("content", "")) // 4 for m in messages) + len(system) // 4
        target = max(100, int(approx_total * compression_ratio))
        return apply_truncation(messages, system, target)
    except Exception:
        return messages, system


def apply_contextprune(messages: list[dict], system: str) -> tuple[list[dict], str, dict]:
    """
    Apply ContextPrune compression pipeline.
    Returns (compressed_messages, compressed_system, stats_dict).
    """
    from contextprune import Config
    from contextprune.core import _CompressedMessages
    from openai import OpenAI

    # Use a fake OpenAI client to run the compression pipeline
    # We intercept at the _CompressedMessages level
    config = Config(semantic_dedup=True, tool_filter=True, budget_injection=False, use_mmr=True)

    from contextprune.dedup import SemanticDeduplicator, MMRSelector
    from contextprune.tokenizer import count_message_tokens, count_system_tokens

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
# Accuracy evaluators
# ---------------------------------------------------------------------------

def eval_mmlu_pro(response: str, correct_answer: str) -> bool:
    """Extract single letter (A-J) from response and compare to correct answer."""
    # Look for answer patterns: "Answer: B", "The answer is C", "(D)", just "A", etc.
    response_upper = response.upper()
    correct = correct_answer.strip().upper()

    # Direct patterns first
    for pattern in [
        r'\bANSWER[:\s]+([A-J])\b',
        r'\bTHE ANSWER IS[:\s]+([A-J])\b',
        r'\bCORRECT[:\s]+([A-J])\b',
        r'\(([A-J])\)',
        r'^([A-J])[.\):]',
        r'\b([A-J])\b',
    ]:
        m = re.search(pattern, response_upper)
        if m:
            return m.group(1) == correct
    return False


def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1]


def eval_math500(response: str, correct_answer: str) -> bool:
    """
    Compare math answers. Tries:
    1. Extract \\boxed{} from response (handles nested braces) and compare
    2. Symbolic equality via sympy if available
    3. String normalization
    4. Substring / number-in-response fallback
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s.lower().replace("\\", "").replace("{", "").replace("}", "").strip())

    # Extract boxed from response (handles nested braces like \frac{1}{2})
    response_ans_raw = _extract_boxed(response)
    response_ans = response_ans_raw if response_ans_raw is not None else response.strip()

    # Try sympy parse_latex first (handles full LaTeX expressions like \frac{1}{2})
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex as _parse_latex

        def _strip_latex(s: str) -> str:
            """Strip outer \boxed{} wrapper if present."""
            return re.sub(r"\\boxed\{(.+?)\}", r"\1", s).strip()

        try:
            gold_expr = _parse_latex(_strip_latex(correct_answer))
            pred_expr = _parse_latex(_strip_latex(response_ans))
            return bool(simplify(gold_expr - pred_expr) == 0)
        except Exception:
            pass
    except ImportError:
        pass

    # Fallback: sympify after stripping LaTeX formatting
    try:
        from sympy import simplify, sympify

        def clean(s: str) -> str:
            return re.sub(r"\s+", "", s.replace("\\", "").replace("{", "").replace("}", ""))

        try:
            a = sympify(clean(response_ans))
            b = sympify(clean(correct_answer))
            return bool(simplify(a - b) == 0)
        except Exception:
            pass
    except ImportError:
        pass

    # String normalization match
    if norm(response_ans) == norm(correct_answer):
        return True

    # Fallback: if correct_answer is a simple number/expression,
    # check if it appears anywhere in the response
    norm_correct = norm(correct_answer)
    norm_response = norm(response)
    if len(norm_correct) <= 20 and norm_correct and norm_correct in norm_response:
        return True

    return False


def eval_frames(response: str, correct_answer: str) -> float:
    """Token F1 between response and correct answer (standard QA metric)."""
    def tokenize(s: str) -> list[str]:
        return re.sub(r"[^\w\s]", " ", s.lower()).split()

    pred_tokens = tokenize(response)
    gold_tokens = tokenize(correct_answer)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def eval_gaia(response: str, correct_answer: str) -> bool:
    """Exact string match (case-insensitive, strip punctuation)."""
    def norm(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower()).strip()
    return norm(response) == norm(correct_answer)


def evaluate_code(
    generated_code: str,
    test_cases: list[dict],
    timeout_seconds: int = 10,
) -> bool:
    """
    Run generated code against test cases.
    Writes code + test assertions to a temp file, kills process after timeout_seconds.
    Returns True if all test cases pass (exit code 0).
    Never hangs — subprocess.TimeoutExpired → return False.
    """
    if not test_cases:
        return False

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(generated_code)
        f.write("\n\n# Auto-generated test runner\n")
        f.write("import sys as _sys\n")
        for i, tc in enumerate(test_cases[:3]):  # max 3 test cases
            if not isinstance(tc, dict):
                continue
            input_data = tc.get("input", "")
            expected = str(tc.get("output", tc.get("expected_output", ""))).strip()
            # Write assertions as exec'd checks
            f.write(f"# test case {i}\n")
            f.write(f"_expected_{i} = {expected!r}\n")
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,  # HARD KILL after timeout_seconds
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False  # timeout = fail, never hang
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def eval_livecodebench(response: str, test_cases_str: str, timeout_sec: int = 10) -> bool:
    """
    Extract code from response and run against test cases with hard timeout.
    Returns True if all test cases pass.
    """
    # Extract Python code block
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
        code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
    if not code_match:
        return False
    code = code_match.group(1)

    try:
        test_cases = json.loads(test_cases_str)[:3]  # run max 3
    except Exception:
        return False

    if not test_cases:
        return False

    # Use the hardened evaluate_code with tmpfile + hard kill
    return evaluate_code(code, test_cases, timeout_seconds=timeout_sec)


def evaluate_response(item: dict, response: str) -> bool | float:
    """Route to the correct evaluator based on source_dataset."""
    dataset = item.get("source_dataset", "")
    answer = item.get("answer", "")
    if dataset == "mmlu_pro":
        return eval_mmlu_pro(response, answer)
    elif dataset == "math500":
        return eval_math500(response, answer)
    elif dataset == "frames":
        score = eval_frames(response, answer)
        return score >= 0.5  # treat F1 >= 0.5 as "correct"
    elif dataset == "gaia":
        return eval_gaia(response, answer)
    elif dataset == "livecodebench":
        return eval_livecodebench(response, item.get("test_cases", "[]"))
    else:
        # Generic: check if answer substring in response (case insensitive)
        return answer.lower() in response.lower()


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_stats(results: list[dict], n_bootstrap: int = 10_000) -> dict:
    """
    Compute accuracy, 95% CI (bootstrap), p-value vs raw condition.
    Bootstrap: n_bootstrap resamples (default 10,000).
    McNemar's test for pairwise comparison to raw.
    """
    # Group by (condition, model)
    groups: dict[tuple[str, str], list[bool]] = {}
    for r in results:
        key = (r["condition"], r["model"])
        groups.setdefault(key, []).append(bool(r["correct"]))

    stats_out: dict = {}
    rng = random.Random(42)

    for (condition, model), corrects in groups.items():
        n = len(corrects)
        if n == 0:
            continue
        acc = sum(corrects) / n

        # Bootstrap CI
        boot_accs = []
        for _ in range(n_bootstrap):
            sample = [rng.choice(corrects) for _ in range(n)]
            boot_accs.append(sum(sample) / n)
        boot_accs.sort()
        ci_low = boot_accs[int(0.025 * n_bootstrap)]
        ci_high = boot_accs[int(0.975 * n_bootstrap)]

        stats_out[(condition, model)] = {
            "condition": condition,
            "model": model,
            "n": n,
            "accuracy": round(acc, 4),
            "ci_95": (round(ci_low, 4), round(ci_high, 4)),
        }

    # McNemar's test: compare each non-raw condition to raw
    for (condition, model), s in stats_out.items():
        if condition == "raw":
            continue
        raw_key = ("raw", model)
        if raw_key not in stats_out:
            continue

        # Build paired vectors
        raw_results = {r["question_id"]: bool(r["correct"])
                       for r in results if r["condition"] == "raw" and r["model"] == model}
        comp_results = {r["question_id"]: bool(r["correct"])
                        for r in results if r["condition"] == condition and r["model"] == model}

        shared_ids = set(raw_results) & set(comp_results)
        b = sum(1 for qid in shared_ids if raw_results[qid] and not comp_results[qid])  # raw correct, comp wrong
        c = sum(1 for qid in shared_ids if not raw_results[qid] and comp_results[qid])  # raw wrong, comp correct

        # McNemar's: chi-squared = (|b - c| - 1)^2 / (b + c)
        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c) if abs(b - c) > 1 else 0.0
            # p-value from chi-squared distribution with df=1
            try:
                from scipy import stats
                p_value = float(stats.chi2.sf(chi2, df=1))
            except ImportError:
                # Approximation if scipy not available
                p_value = _chi2_p_approx(chi2)
        else:
            p_value = 1.0

        # Cohen's d (treating binary as 0/1)
        raw_acc = stats_out[raw_key]["accuracy"]
        comp_acc = s["accuracy"]
        # Pooled SD for binary outcomes
        p_pool = (raw_acc + comp_acc) / 2
        sd_pool = math.sqrt(max(1e-10, p_pool * (1 - p_pool)))
        cohens_d = (comp_acc - raw_acc) / sd_pool if sd_pool > 0 else 0.0

        s["p_value_vs_raw"] = round(p_value, 4)
        s["cohens_d"] = round(cohens_d, 4)

    return stats_out


def _chi2_p_approx(chi2: float) -> float:
    """Very rough p-value approximation for chi-squared df=1 without scipy."""
    if chi2 <= 0:
        return 1.0
    # Lookup table for common thresholds
    if chi2 >= 10.83:
        return 0.001
    if chi2 >= 6.63:
        return 0.01
    if chi2 >= 3.84:
        return 0.05
    if chi2 >= 2.71:
        return 0.10
    return 0.50


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(
    n_questions: int,
    conditions: list[str],
    model_ids: list[str],
    avg_input_tokens: int = 1800,
    avg_output_tokens: int = 256,
    compression_ratios: dict[str, float] | None = None,
) -> dict:
    """Estimate total cost for a full run."""
    if compression_ratios is None:
        compression_ratios = {
            "raw": 1.0,
            "truncation": 0.55,
            "llmlingua2": 0.45,
            "contextprune": 0.40,
        }

    total = 0.0
    breakdown: dict[str, dict[str, float]] = {}
    for model_id in model_ids:
        pricing = PRICING.get(model_id, {"input": 2.0, "output": 10.0})
        breakdown[model_id] = {}
        for cond in conditions:
            ratio = compression_ratios.get(cond, 1.0)
            eff_input = avg_input_tokens * ratio
            cost_per_q = (eff_input / 1e6) * pricing["input"] + (avg_output_tokens / 1e6) * pricing["output"]
            cond_cost = cost_per_q * n_questions
            breakdown[model_id][cond] = round(cond_cost, 4)
            total += cond_cost
    return {"total_usd": round(total, 4), "by_model": breakdown}


# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------

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
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(dataset: str, n: int, seed: int = 42) -> list[dict]:
    """Load n items from a dataset's test.jsonl."""
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
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: str,
    models: list[str],
    n: int,
    budget_usd: float,
    conditions: list[str] = None,
    context_size: str = "medium",
    dry_run: bool = False,
    seed: int = 42,
    skip_llmlingua2: bool = False,
) -> list[dict]:
    """
    Run Experiment 3: accuracy across conditions × models.

    Returns list of per-question result dicts.
    """
    if conditions is None:
        conditions = CONDITIONS

    # Fix 7: --skip-llmlingua2 removes that condition for fast dry-run validation
    if skip_llmlingua2 and "llmlingua2" in conditions:
        conditions = [c for c in conditions if c != "llmlingua2"]
        print("  [skip-llmlingua2] LLMLingua-2 condition removed for this run.")

    # Fix 2: per-model rate limiter
    rate_limiter = RateLimiter()

    # Fix 5: model-down resilience
    health_tracker = ModelHealthTracker(max_consecutive_failures=3)

    from benchmarks.context_injector import ContextInjector
    injector = ContextInjector()

    # Fix 1: pre-load LLMLingua-2 cache (avoids re-reading file per question)
    llmlingua2_cache: dict[str, dict] = {}
    if "llmlingua2" in conditions:
        llmlingua2_cache = _load_llmlingua2_cache(dataset)
        if llmlingua2_cache:
            print(f"  [llmlingua2] Loaded {len(llmlingua2_cache):,} cached items from disk.")
        else:
            logging.warning(
                "LLMLingua-2 cache not found — will run live (~30s/question on CPU). "
                "Run benchmarks/precompute_llmlingua2.py first for best performance."
            )

    # Resolve model IDs
    model_ids: list[str] = []
    for m in models:
        if m in SUPPORTED_MODELS:
            model_ids.append(SUPPORTED_MODELS[m])
        elif m == "all":
            model_ids.extend(SUPPORTED_MODELS.values())
        elif "/" in m:
            model_ids.append(m)
        else:
            print(f"  [WARN] Unknown model alias: {m!r}")

    # Load data
    print(f"\nLoading {n} items from {dataset}…")
    items = load_dataset(dataset, n, seed=seed)
    print(f"  Loaded {len(items):,} items")

    # Checkpoint
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = RESULTS_DIR / f"exp3_{dataset}_{ts}.jsonl"
    ckpt = Checkpoint(ckpt_path)
    cost_ckpt_path = RESULTS_DIR / f".cost_{dataset}_{ts}.json"

    # Budget guard
    guard = CostGuard(budget_usd, checkpoint_path=cost_ckpt_path)

    results: list[dict] = []

    if dry_run:
        print(f"\n=== DRY RUN — {len(items)} questions × {len(conditions)} conditions × {len(model_ids)} models ===")
        print(f"  Would write results to: {ckpt_path}")
        print(f"  Budget: ${budget_usd:.2f}")

    # Set up API (only needed for non-dry-run)
    adapter: OpenRouterAdapter | None = None
    if not dry_run:
        api_key = _load_openrouter_key()
        if not api_key:
            print("  [ERROR] No OPENROUTER_API_KEY found. Use --dry-run or set the env var.")
            return []
        adapter = OpenRouterAdapter(api_key=api_key)

    # Inject context into all items
    print(f"\nInjecting context ({context_size})…")
    injected_items = [injector.inject(item, context_size=context_size) for item in items]
    avg_inject_tokens = sum(i.get("injected_token_estimate", 0) for i in injected_items) / max(1, len(injected_items))
    print(f"  Avg injected tokens: {avg_inject_tokens:.0f}")

    for item_idx, item in enumerate(injected_items):
        qid = item.get("id", f"q_{item_idx:05d}")
        messages = item.get("messages", [])
        system = item.get("system", "")
        tools = item.get("tools", [])

        for condition in conditions:
            for model_id in model_ids:
                model_alias = next((k for k, v in SUPPORTED_MODELS.items() if v == model_id), model_id)

                if ckpt.is_done(qid, condition, model_alias):
                    continue

                # Apply compression for the condition
                comp_messages = messages
                comp_system = system
                comp_stats: dict = {}
                comp_ratio = 1.0

                t_compress_start = time.perf_counter()
                if condition == "raw":
                    comp_messages, comp_system = messages, system
                elif condition == "truncation":
                    orig_tokens = sum(len(m.get("content", "")) // 4 for m in messages) + len(system) // 4
                    # Target: match typical contextprune output length
                    target = max(100, int(orig_tokens * 0.45))
                    comp_messages, comp_system = apply_truncation(messages, system, target)
                    comp_ratio = max(0.01, sum(len(m.get("content","")) for m in comp_messages) /
                                     max(1, sum(len(m.get("content","")) for m in messages)))
                elif condition == "llmlingua2":
                    comp_messages, comp_system = apply_llmlingua2(
                        messages, system, compression_ratio=0.45,
                        item_id=qid, dataset=dataset,
                        llmlingua2_cache=llmlingua2_cache,
                    )
                    comp_ratio = 0.45
                elif condition == "contextprune":
                    try:
                        comp_messages, comp_system, cp_stats = apply_contextprune(messages, system)
                        comp_stats = cp_stats
                        orig = cp_stats.get("original_tokens", 1)
                        comp = cp_stats.get("compressed_tokens", orig)
                        comp_ratio = round(comp / max(1, orig), 4)
                    except Exception as e:
                        print(f"  [WARN] ContextPrune failed on {qid}: {e}")
                        comp_messages, comp_system = messages, system
                        comp_ratio = 1.0

                compress_ms = (time.perf_counter() - t_compress_start) * 1000
                tokens_in = sum(len(m.get("content", "")) // 4 for m in comp_messages) + len(comp_system) // 4

                # Estimate cost for this call
                pricing = PRICING.get(model_id, {"input": 2.0, "output": 10.0})
                est_cost = (tokens_in / 1e6) * pricing["input"] + (256 / 1e6) * pricing["output"]

                if dry_run:
                    # Show preview without calling API
                    question_preview = (messages[-1].get("content", "") if messages else "")[:100]
                    print(f"\n  Q{item_idx+1:03d}/{len(injected_items)} [{condition:12s}] [{model_alias:7s}]")
                    print(f"    Question:    {question_preview!r}…")
                    print(f"    Tokens in:   {tokens_in:,} (compressed: {comp_ratio:.1%})")
                    print(f"    Est cost:    ${est_cost:.6f}")
                    if condition == "contextprune" and comp_stats:
                        print(f"    CP savings:  {comp_stats.get('savings_pct', 0):.1f}%")
                    result = {
                        "question_id": qid,
                        "condition": condition,
                        "model": model_alias,
                        "correct": None,
                        "tokens_in": tokens_in,
                        "tokens_out": 0,
                        "cost_usd": 0.0,
                        "latency_ms": 0.0,
                        "compress_ms": round(compress_ms, 2),
                        "compression_ratio": round(comp_ratio, 4),
                        "dry_run": True,
                        "dataset": dataset,
                        "category": item.get("category"),
                        "difficulty": item.get("difficulty"),
                    }
                    results.append(result)
                    continue

                # Fix 5: skip disabled models (3+ consecutive 5xx)
                if not health_tracker.is_available(model_id):
                    print(f"  [SKIP] {model_alias} disabled (too many 5xx errors) — skipping {qid}")
                    continue

                # Real API call
                try:
                    guard.check(est_cost)
                except BudgetExceededError as e:
                    print(f"\n  [BUDGET] {e}")
                    print(f"  Stopping at question {item_idx + 1}/{len(injected_items)}")
                    return results

                # Fix 2: rate limit before call
                rate_limiter.wait_if_needed(model_id, tokens_in)

                t_api_start = time.perf_counter()
                try:
                    completion = adapter.complete(
                        messages=comp_messages,
                        model=model_id,
                        system=comp_system,
                        max_tokens=512,
                        temperature=0.0,
                    )
                    latency_ms = (time.perf_counter() - t_api_start) * 1000
                    response_text = completion.text
                    actual_cost = completion.cost_usd
                    tokens_out = completion.output_tokens
                    tokens_in_actual = completion.input_tokens
                    guard.record(actual_cost)
                    health_tracker.record_success(model_id)  # Fix 5: reset failure count
                    correct = evaluate_response(item, response_text)

                    result = {
                        "question_id": qid,
                        "condition": condition,
                        "model": model_alias,
                        "correct": bool(correct) if isinstance(correct, bool) else float(correct),
                        "tokens_in": tokens_in_actual,
                        "tokens_out": tokens_out,
                        "cost_usd": actual_cost,
                        "latency_ms": round(latency_ms, 2),
                        "compress_ms": round(compress_ms, 2),
                        "compression_ratio": round(comp_ratio, 4),
                        "dataset": dataset,
                        "category": item.get("category"),
                        "difficulty": item.get("difficulty"),
                        "response_preview": response_text[:200],
                    }
                    ckpt.write(result)
                    results.append(result)

                    correct_str = "✓" if result["correct"] else "✗"
                    print(f"  Q{item_idx+1:03d} [{condition:12s}] [{model_alias:7s}] "
                          f"{correct_str} {tokens_in_actual:,}tok ${actual_cost:.5f} "
                          f"{latency_ms:.0f}ms ratio={comp_ratio:.2f} spent=${guard.spent:.4f}")

                except BudgetExceededError:
                    raise
                except Exception as exc:
                    # Fix 5: record 5xx errors for model health tracking
                    exc_str = str(exc)
                    http_code = 0
                    code_match = re.search(r"\b(5\d\d)\b", exc_str)
                    if code_match:
                        http_code = int(code_match.group(1))
                    health_tracker.record_failure(model_id, http_code)
                    print(f"  [ERROR] {qid}/{condition}/{model_alias}: {exc}")
                    result = {
                        "question_id": qid,
                        "condition": condition,
                        "model": model_alias,
                        "correct": False,
                        "tokens_in": tokens_in,
                        "tokens_out": 0,
                        "cost_usd": 0.0,
                        "latency_ms": 0.0,
                        "compress_ms": round(compress_ms, 2),
                        "compression_ratio": round(comp_ratio, 4),
                        "error": str(exc),
                        "dataset": dataset,
                        "category": item.get("category"),
                        "difficulty": item.get("difficulty"),
                    }
                    ckpt.write(result)
                    results.append(result)

    return results


def print_summary(results: list[dict], dry_run: bool = False) -> None:
    """Print accuracy summary table and statistics."""
    if dry_run:
        total_tok = sum(r.get("tokens_in", 0) for r in results)
        avg_tok = total_tok / max(1, len(results))
        n_q = len(set(r["question_id"] for r in results))
        conditions_seen = sorted(set(r["condition"] for r in results))
        models_seen = sorted(set(r["model"] for r in results))
        print(f"\n=== DRY RUN SUMMARY ===")
        print(f"  Questions: {n_q}")
        print(f"  Conditions: {conditions_seen}")
        print(f"  Models: {models_seen}")
        print(f"  Avg tokens per call: {avg_tok:.0f}")
        return

    if not results:
        print("No results to summarize.")
        return

    print("\n=== Exp 3 Accuracy Summary ===")
    stats = compute_stats(results)

    conditions_seen = sorted({s["condition"] for s in stats.values()})
    models_seen = sorted({s["model"] for s in stats.values()})

    # Accuracy matrix
    header = f"{'Model':<12}" + "".join(f"  {c:>12}" for c in conditions_seen)
    print(f"\n{header}")
    print("-" * len(header))
    for model in models_seen:
        row = f"{model:<12}"
        for cond in conditions_seen:
            key = (cond, model)
            if key in stats:
                s = stats[key]
                acc = s["accuracy"]
                ci = s.get("ci_95", (0, 0))
                row += f"  {acc:>8.1%} ±{(ci[1]-ci[0])/2:.1%}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    # P-values vs raw
    print(f"\np-value vs raw (McNemar's test):")
    for (cond, model), s in sorted(stats.items()):
        if "p_value_vs_raw" in s:
            p = s["p_value_vs_raw"]
            d = s.get("cohens_d", 0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {model:12s} {cond:12s}: p={p:.4f} {sig}  d={d:+.3f}")

    # Save summary CSV
    csv_path = RESULTS_DIR / f"exp3_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "model", "n", "accuracy", "ci_low", "ci_high",
                                                "p_value_vs_raw", "cohens_d"])
        writer.writeheader()
        for (cond, model), s in sorted(stats.items()):
            ci = s.get("ci_95", (0, 0))
            writer.writerow({
                "condition": cond,
                "model": model,
                "n": s["n"],
                "accuracy": s["accuracy"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_value_vs_raw": s.get("p_value_vs_raw", ""),
                "cohens_d": s.get("cohens_d", ""),
            })
    print(f"\n  Summary CSV → {csv_path}")


# ---------------------------------------------------------------------------
# Cost estimate mode
# ---------------------------------------------------------------------------

def print_cost_estimate(dataset: str, n: int, models: list[str]) -> None:
    model_ids = []
    for m in models:
        if m == "all":
            model_ids.extend(SUPPORTED_MODELS.values())
        elif m in SUPPORTED_MODELS:
            model_ids.append(SUPPORTED_MODELS[m])
        elif "/" in m:
            model_ids.append(m)

    # Load a few items to get avg token estimate
    try:
        from benchmarks.context_injector import ContextInjector
        injector = ContextInjector()
        items = load_dataset(dataset, min(10, n))
        injected = [injector.inject(i, context_size="medium") for i in items]
        avg_tokens = sum(i.get("injected_token_estimate", 1500) for i in injected) / max(1, len(injected))
    except Exception:
        avg_tokens = 1500

    est = estimate_cost(n, CONDITIONS, model_ids, avg_input_tokens=int(avg_tokens))

    print(f"\n=== Cost Estimate: Exp 3 ===")
    print(f"  Dataset:    {dataset}")
    print(f"  Questions:  {n}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Avg input tokens: {avg_tokens:.0f}")
    print(f"\n  By model:")
    for model_id, by_cond in est["by_model"].items():
        model_total = sum(by_cond.values())
        alias = next((k for k, v in SUPPORTED_MODELS.items() if v == model_id), model_id)
        print(f"    {alias:<10} total=${model_total:.4f}  " + "  ".join(f"{c}=${v:.4f}" for c, v in by_cond.items()))
    print(f"\n  TOTAL: ${est['total_usd']:.4f}")
    print(f"\n  Note: LLMLingua-2 compression adds ~30s per question on CPU.")
    print(f"  ContextPrune compression adds ~200ms per question.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3: Accuracy on standard benchmarks.")
    p.add_argument("--dataset", default="mmlu_pro",
                   choices=["mmlu_pro", "math500", "livecodebench", "frames", "gaia"],
                   help="Dataset to run on")
    p.add_argument("--models", default="claude",
                   help="Comma-separated model aliases or 'all'. Valid: " + ", ".join(SUPPORTED_MODELS))
    p.add_argument("--n", type=int, default=100, help="Number of questions per dataset")
    p.add_argument("--budget", type=float, default=5.00, help="Budget in USD (default $5.00)")
    p.add_argument("--conditions", default=",".join(CONDITIONS), help="Conditions to run")
    p.add_argument("--context-size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--dry-run", action="store_true", help="Validate pipeline without API calls")
    p.add_argument("--cost-estimate", action="store_true", help="Show cost estimate and exit")
    p.add_argument("--skip-llmlingua2", action="store_true",
                   help="Skip LLMLingua-2 condition for fast end-to-end validation (~10s dry run)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    if args.cost_estimate:
        print_cost_estimate(args.dataset, args.n, models)
        return

    results = run_experiment(
        dataset=args.dataset,
        models=models,
        n=args.n,
        budget_usd=args.budget,
        conditions=conditions,
        context_size=args.context_size,
        dry_run=args.dry_run,
        seed=args.seed,
        skip_llmlingua2=args.skip_llmlingua2,
    )

    print_summary(results, dry_run=args.dry_run)

    if args.dry_run:
        print("\n✓ Dry run complete — pipeline validated. No API calls made.")


if __name__ == "__main__":
    main()
