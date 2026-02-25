"""
Experiment 5: API Accuracy (optional — requires API key)

Runs 20 factual/math/code/tool questions through the model with and without
contextprune, then compares accuracy.

Supports multiple models via OpenRouter. Pass --model <alias> to select one
model, or --model all to run all 4 and produce a comparison table.

If no API key is found, the experiment is skipped with a clear note.

Usage:
    python benchmarks/exp5_api_accuracy.py                  # default: claude
    python benchmarks/exp5_api_accuracy.py --model gemini
    python benchmarks/exp5_api_accuracy.py --model all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/home/keith/contextprune")

from contextprune.adapters import SUPPORTED_MODELS, OpenRouterAdapter


def _load_openrouter_key() -> Optional[str]:
    """Try to load OpenRouter API key from environment or 1Password."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key

    try:
        result = subprocess.run(
            [
                "bash",
                "-c",
                "OP_SERVICE_ACCOUNT_TOKEN=$(cat /etc/op-service-account-token) "
                "op item get 'OPENROUTER_API_KEY' --vault 'Keef Secrets' "
                "--fields credential --reveal 2>/dev/null",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        key = result.stdout.strip()
        if key and key not in ("None", "", "null"):
            return key
    except Exception:
        pass

    return None


# 20 test questions: 5 factual, 5 math, 5 code, 5 tool selection
TEST_QUESTIONS: List[Dict[str, Any]] = [
    # Factual (5)
    {
        "category": "factual",
        "question": "What is the capital of France?",
        "expected": "Paris",
        "check": lambda r: "paris" in r.lower(),
    },
    {
        "category": "factual",
        "question": "Which year was Python first released publicly?",
        "expected": "1991",
        "check": lambda r: "1991" in r,
    },
    {
        "category": "factual",
        "question": "What does REST stand for in web APIs?",
        "expected": "Representational State Transfer",
        "check": lambda r: "representational" in r.lower() and "state" in r.lower(),
    },
    {
        "category": "factual",
        "question": "What is the default port for PostgreSQL?",
        "expected": "5432",
        "check": lambda r: "5432" in r,
    },
    {
        "category": "factual",
        "question": "What HTTP status code means 'Not Found'?",
        "expected": "404",
        "check": lambda r: "404" in r,
    },
    # Math (5)
    {
        "category": "math",
        "question": "What is 17 × 23?",
        "expected": "391",
        "check": lambda r: "391" in r,
    },
    {
        "category": "math",
        "question": "What is the square root of 144?",
        "expected": "12",
        "check": lambda r: "12" in r,
    },
    {
        "category": "math",
        "question": "If a function f(x) = 3x² + 2x - 1, what is f(2)?",
        "expected": "15",
        "check": lambda r: "15" in r,
    },
    {
        "category": "math",
        "question": "What is 2^10?",
        "expected": "1024",
        "check": lambda r: "1024" in r,
    },
    {
        "category": "math",
        "question": "What is 15% of 240?",
        "expected": "36",
        "check": lambda r: "36" in r,
    },
    # Code (5)
    {
        "category": "code",
        "question": "In Python, what does `list(range(3))` return?",
        "expected": "[0, 1, 2]",
        "check": lambda r: "0" in r and "1" in r and "2" in r and "3" not in r.replace("range(3)", ""),
    },
    {
        "category": "code",
        "question": "What is the time complexity of binary search?",
        "expected": "O(log n)",
        "check": lambda r: "log" in r.lower() and ("o(" in r.lower() or "o (" in r.lower()),
    },
    {
        "category": "code",
        "question": "In Python, what does `'hello'[::-1]` evaluate to?",
        "expected": "olleh",
        "check": lambda r: "olleh" in r,
    },
    {
        "category": "code",
        "question": "What Python decorator is used to define a classmethod?",
        "expected": "@classmethod",
        "check": lambda r: "classmethod" in r.lower(),
    },
    {
        "category": "code",
        "question": "What keyword in Python is used to define a generator function?",
        "expected": "yield",
        "check": lambda r: "yield" in r.lower(),
    },
    # Tool selection (5)
    {
        "category": "tool_selection",
        "question": "I need to fetch live stock prices from an external financial API. Which approach is most appropriate: web scraping, a REST API call, or a database query?",
        "expected": "REST API",
        "check": lambda r: "api" in r.lower() or "rest" in r.lower(),
    },
    {
        "category": "tool_selection",
        "question": "A user wants to search for information about climate change. What tool should an AI agent use?",
        "expected": "web search",
        "check": lambda r: "search" in r.lower() or "web" in r.lower(),
    },
    {
        "category": "tool_selection",
        "question": "An agent needs to run a Python script to process data. What should it use?",
        "expected": "code execution",
        "check": lambda r: "execut" in r.lower() or "run" in r.lower() or "code" in r.lower(),
    },
    {
        "category": "tool_selection",
        "question": "A user asks: 'What's on my calendar for tomorrow?' What tool is needed?",
        "expected": "calendar",
        "check": lambda r: "calendar" in r.lower(),
    },
    {
        "category": "tool_selection",
        "question": "To look up a customer's order history, an agent should use which tool type?",
        "expected": "database query",
        "check": lambda r: "database" in r.lower() or "db" in r.lower() or "sql" in r.lower() or "query" in r.lower(),
    },
]

SYSTEM_FOR_TEST = "You are a helpful AI assistant. Answer questions concisely and accurately."


def _run_single_model(
    adapter: OpenRouterAdapter,
    model_alias: str,
) -> Dict[str, Any]:
    """Run 20 questions with and without contextprune for one model."""
    from contextprune import Config, wrap_openai
    from openai import OpenAI

    model_id = SUPPORTED_MODELS.get(model_alias, model_alias)

    # Build a raw OpenAI client pointing at OpenRouter
    raw_openai = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=adapter.client.api_key,
    )
    compressed_openai = wrap_openai(
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=adapter.client.api_key,
        ),
        config=Config(semantic_dedup=True, tool_filter=True, budget_injection=True),
    )

    results = []
    total_tokens_without = 0
    total_tokens_with = 0
    total_cost_without = 0.0
    total_cost_with = 0.0

    for q in TEST_QUESTIONS:
        msgs = [
            {"role": "system", "content": SYSTEM_FOR_TEST},
            {"role": "user", "content": q["question"]},
        ]

        # Without contextprune (direct OpenRouter via adapter)
        try:
            raw_result = adapter.complete(
                messages=[{"role": "user", "content": q["question"]}],
                model=model_alias,
                system=SYSTEM_FOR_TEST,
                max_tokens=150,
                temperature=0.0,
            )
            raw_answer = raw_result.text
            raw_correct = q["check"](raw_answer)
            total_tokens_without += raw_result.input_tokens + raw_result.output_tokens
            total_cost_without += raw_result.cost_usd
        except Exception as e:
            raw_answer = f"ERROR: {e}"
            raw_correct = False

        # With contextprune (wrapped OpenAI client → OpenRouter)
        try:
            comp_resp = compressed_openai.chat.completions.create(
                model=model_id,
                max_tokens=150,
                messages=msgs,
            )
            comp_answer = comp_resp.choices[0].message.content or ""
            comp_correct = q["check"](comp_answer)
            if comp_resp.usage:
                toks = comp_resp.usage.prompt_tokens + comp_resp.usage.completion_tokens
                total_tokens_with += toks
                total_cost_with += adapter._calculate_cost(
                    model_id,
                    comp_resp.usage.prompt_tokens,
                    comp_resp.usage.completion_tokens,
                )
        except Exception as e:
            comp_answer = f"ERROR: {e}"
            comp_correct = False

        results.append({
            "category": q["category"],
            "question": q["question"],
            "expected": q["expected"],
            "raw_answer": raw_answer[:200],
            "raw_correct": raw_correct,
            "comp_answer": comp_answer[:200],
            "comp_correct": comp_correct,
            "agreement": raw_correct == comp_correct,
        })

    raw_acc = sum(1 for r in results if r["raw_correct"]) / len(results)
    comp_acc = sum(1 for r in results if r["comp_correct"]) / len(results)

    return {
        "skipped": False,
        "model_alias": model_alias,
        "model_id": model_id,
        "total_questions": len(results),
        "raw_accuracy": round(raw_acc, 3),
        "comp_accuracy": round(comp_acc, 3),
        "accuracy_delta": round(comp_acc - raw_acc, 3),
        "tokens_without": total_tokens_without,
        "tokens_with": total_tokens_with,
        "cost_without": round(total_cost_without, 6),
        "cost_with": round(total_cost_with, 6),
        "results": results,
    }


def run_exp5(model: str = "claude") -> Dict[str, Any]:
    """Run Experiment 5 — API accuracy comparison via OpenRouter."""
    api_key = _load_openrouter_key()
    if not api_key:
        return {
            "skipped": True,
            "reason": "No OPENROUTER_API_KEY found (checked environment and 1Password).",
            "model": model,
            "results": [],
        }

    adapter = OpenRouterAdapter(api_key=api_key)

    if model == "all":
        model_results = {}
        for alias in SUPPORTED_MODELS:
            print(f"  Running {alias}…", flush=True)
            try:
                model_results[alias] = _run_single_model(adapter, alias)
            except Exception as exc:
                model_results[alias] = {
                    "skipped": True,
                    "reason": str(exc),
                    "model_alias": alias,
                    "model_id": SUPPORTED_MODELS[alias],
                }
        return {
            "skipped": False,
            "mode": "all",
            "models": model_results,
        }

    return _run_single_model(adapter, model)


def print_results(exp5: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: API Accuracy (via OpenRouter)")
    print("=" * 80)

    if exp5.get("skipped"):
        print(f"\nSKIPPED: {exp5['reason']}")
        return

    if exp5.get("mode") == "all":
        # Multi-model comparison table
        print("\nComparison across all 4 models (raw / contextpruned):\n")
        header = f"{'Model':<12} {'Acc (raw)':>10} {'Acc (cp)':>10} {'Δ Acc':>8} {'Tok (raw)':>12} {'Tok (cp)':>12} {'Cost (raw)':>12} {'Cost (cp)':>12}"
        print(header)
        print("-" * len(header))
        for alias, res in exp5["models"].items():
            if res.get("skipped"):
                print(f"{alias:<12} {'SKIPPED: ' + res.get('reason', '')[:60]}")
                continue
            print(
                f"{alias:<12} "
                f"{res['raw_accuracy']:>10.1%} "
                f"{res['comp_accuracy']:>10.1%} "
                f"{res['accuracy_delta']:>+8.1%} "
                f"{res['tokens_without']:>12,} "
                f"{res['tokens_with']:>12,} "
                f"{res['cost_without']:>12.6f} "
                f"{res['cost_with']:>12.6f}"
            )
        return

    # Single model
    print(f"\nModel: {exp5['model_alias']} ({exp5['model_id']})")
    print(f"Raw accuracy:        {exp5['raw_accuracy']:.1%}")
    print(f"Compressed accuracy: {exp5['comp_accuracy']:.1%}")
    print(f"Delta:               {exp5['accuracy_delta']:+.1%}")
    print(f"Tokens (raw/cp):     {exp5['tokens_without']:,} / {exp5['tokens_with']:,}")
    print(f"Cost (raw/cp):       ${exp5['cost_without']:.6f} / ${exp5['cost_with']:.6f}")
    print()
    print(f"{'Category':<15} {'Q':<50} {'Raw':>5} {'Comp':>5}")
    print("-" * 80)
    for r in exp5["results"]:
        raw_mark = "✓" if r["raw_correct"] else "✗"
        comp_mark = "✓" if r["comp_correct"] else "✗"
        print(f"{r['category']:<15} {r['question'][:49]:<50} {raw_mark:>5} {comp_mark:>5}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 5: API Accuracy via OpenRouter multi-model."
    )
    parser.add_argument(
        "--model",
        default="claude",
        help=(
            'Model alias or "all". Valid aliases: '
            + ", ".join(SUPPORTED_MODELS.keys())
            + '. Default: claude'
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_exp5(model=args.model)
    print_results(result)
