"""
Experiment 5: API Accuracy (optional — requires API key)

Runs 20 factual/math/code/tool questions through the model with and without
contextprune, then compares accuracy.

If no API key is found, the experiment is skipped with a clear note.
"""

from __future__ import annotations

import os
import sys
import subprocess
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/home/keith/contextprune")


def _load_api_keys() -> Tuple[Optional[str], Optional[str]]:
    """Try to load API keys from environment, then from op-secret."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Always try op-secret to get the freshest keys
    try:
        result = subprocess.run(
            ["bash", "-c", "eval $(op-secret env) 2>/dev/null && printf 'ANTHROPIC_API_KEY=%s\nOPENAI_API_KEY=%s\n' \"$ANTHROPIC_API_KEY\" \"$OPENAI_API_KEY\""],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in result.stdout.splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val and val not in ("None", "", "null"):
                    anthropic_key = val
            if line.startswith("OPENAI_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val and val not in ("None", "", "null"):
                    openai_key = val
    except Exception:
        pass

    return anthropic_key, openai_key


# 20 test questions: 5 factual, 5 math, 5 code, 5 tool selection
TEST_QUESTIONS = [
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


def run_exp5() -> Dict[str, Any]:
    """Run Experiment 5 — API accuracy comparison."""
    anthropic_key, openai_key = _load_api_keys()

    if not anthropic_key and not openai_key:
        return {
            "skipped": True,
            "reason": "No ANTHROPIC_API_KEY or OPENAI_API_KEY found (checked environment and op-secret).",
            "results": [],
        }

    # Prefer Anthropic
    if anthropic_key:
        try:
            return _run_with_anthropic(anthropic_key)
        except ImportError:
            if openai_key:
                pass  # fall through to OpenAI
            else:
                return {
                    "skipped": True,
                    "reason": "anthropic package not installed. Install with: pip install anthropic",
                    "results": [],
                }

    if openai_key:
        try:
            return _run_with_openai(openai_key)
        except ImportError:
            return {
                "skipped": True,
                "reason": "openai package not installed. Install with: pip install openai",
                "results": [],
            }

    return {
        "skipped": True,
        "reason": "No API keys available.",
        "results": [],
    }


def _run_with_anthropic(api_key: str) -> Dict[str, Any]:
    import anthropic
    from contextprune import wrap, Config

    raw_client = anthropic.Anthropic(api_key=api_key)
    compressed_client = wrap(
        anthropic.Anthropic(api_key=api_key),
        config=Config(semantic_dedup=True, tool_filter=True, budget_injection=True),
    )

    results = []
    model = "claude-haiku-4-5"  # fast/cheap for benchmarking

    for q in TEST_QUESTIONS:
        msgs = [{"role": "user", "content": q["question"]}]

        # Without contextprune
        try:
            raw_resp = raw_client.messages.create(
                model=model,
                max_tokens=150,
                system=SYSTEM_FOR_TEST,
                messages=msgs,
            )
            raw_answer = raw_resp.content[0].text if raw_resp.content else ""
            raw_correct = q["check"](raw_answer)
        except Exception as e:
            raw_answer = f"ERROR: {e}"
            raw_correct = False

        # With contextprune
        try:
            comp_resp = compressed_client.messages.create(
                model=model,
                max_tokens=150,
                system=SYSTEM_FOR_TEST,
                messages=msgs,
            )
            comp_answer = comp_resp.content[0].text if comp_resp.content else ""
            comp_correct = q["check"](comp_answer)
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
        "provider": "anthropic",
        "model": model,
        "total_questions": len(results),
        "raw_accuracy": round(raw_acc, 3),
        "comp_accuracy": round(comp_acc, 3),
        "accuracy_delta": round(comp_acc - raw_acc, 3),
        "results": results,
    }


def _run_with_openai(api_key: str) -> Dict[str, Any]:
    import openai
    from contextprune import wrap_openai, Config

    raw_client = openai.OpenAI(api_key=api_key)
    compressed_client = wrap_openai(
        openai.OpenAI(api_key=api_key),
        config=Config(semantic_dedup=True, tool_filter=True, budget_injection=True),
    )

    results = []
    model = "gpt-4.1"

    for q in TEST_QUESTIONS:
        msgs = [
            {"role": "system", "content": SYSTEM_FOR_TEST},
            {"role": "user", "content": q["question"]},
        ]

        try:
            raw_resp = raw_client.chat.completions.create(
                model=model,
                max_tokens=150,
                messages=msgs,
            )
            raw_answer = raw_resp.choices[0].message.content or ""
            raw_correct = q["check"](raw_answer)
        except Exception as e:
            raw_answer = f"ERROR: {e}"
            raw_correct = False

        try:
            comp_resp = compressed_client.chat.completions.create(
                model=model,
                max_tokens=150,
                messages=msgs,
            )
            comp_answer = comp_resp.choices[0].message.content or ""
            comp_correct = q["check"](comp_answer)
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
        "provider": "openai",
        "model": model,
        "total_questions": len(results),
        "raw_accuracy": round(raw_acc, 3),
        "comp_accuracy": round(comp_acc, 3),
        "accuracy_delta": round(comp_acc - raw_acc, 3),
        "results": results,
    }


def print_results(exp5: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: API Accuracy")
    print("=" * 80)
    if exp5.get("skipped"):
        print(f"\nSKIPPED: {exp5['reason']}")
        return

    print(f"\nProvider: {exp5['provider']}, Model: {exp5['model']}")
    print(f"Raw accuracy:        {exp5['raw_accuracy']:.1%}")
    print(f"Compressed accuracy: {exp5['comp_accuracy']:.1%}")
    print(f"Delta:               {exp5['accuracy_delta']:+.1%}")
    print()
    print(f"{'Category':<15} {'Q':<50} {'Raw':>5} {'Comp':>5}")
    print("-" * 80)
    for r in exp5["results"]:
        raw_mark = "✓" if r["raw_correct"] else "✗"
        comp_mark = "✓" if r["comp_correct"] else "✗"
        print(f"{r['category']:<15} {r['question'][:49]:<50} {raw_mark:>5} {comp_mark:>5}")


if __name__ == "__main__":
    result = run_exp5()
    print_results(result)
