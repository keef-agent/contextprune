"""
Download and preprocess all Phase 2 benchmark datasets.
Saves to benchmarks/data/<dataset_name>/

Datasets:
  - mmlu_pro: TIGER-Lab/MMLU-Pro (10-choice, 12K test questions)
  - math500: lighteval/MATH (sample 500 competition math problems)
  - livecodebench: livecodebench/code_generation_lite
  - frames: google/frames-benchmark
  - gaia: gaia-benchmark/GAIA (validation split, level 1+2 only — level 3 too hard)
  - sharegpt: anon8231489123/ShareGPT_Vicuna_unfiltered

Skip for now (require special setup): helmet, swe_bench_verified

Usage:
  python3 benchmarks/data/download_datasets.py           # download all
  python3 benchmarks/data/download_datasets.py --validate # validate existing downloads
  python3 benchmarks/data/download_datasets.py --dataset mmlu_pro
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_DIR = Path(__file__).parent
STATS_FILE = DATA_DIR / "download_stats.json"

DIFFICULTY_MAP = {1: "easy", 2: "easy", 3: "medium", 4: "hard", 5: "hard"}

MMLU_CATEGORY_TO_DOMAIN = {
    "math": "math", "physics": "factual", "chemistry": "factual",
    "biology": "factual", "history": "factual", "law": "factual",
    "economics": "factual", "psychology": "factual", "computer science": "code",
    "engineering": "code", "medicine": "factual", "business": "factual",
    "philosophy": "factual", "other": "factual",
}


def _tok_approx(text: str) -> int:
    """Token count approximation via char/4."""
    return max(1, len(text) // 4)


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items):,} items → {path}")


def download_mmlu_pro(out_dir: Path, n: int = 500, seed: int = 42) -> dict:
    """Download TIGER-Lab/MMLU-Pro and sample n questions stratified by category."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] datasets not installed. Run: pip install datasets")
        return {}

    print("  Downloading TIGER-Lab/MMLU-Pro test split …")
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {}

    # Group by category for stratified sampling
    by_category: dict[str, list] = defaultdict(list)
    for item in ds:
        by_category[item.get("category", "other")].append(item)

    categories = list(by_category.keys())
    per_cat = max(1, n // len(categories))
    rng = random.Random(seed)

    sampled: list[dict] = []
    for cat, items in by_category.items():
        rng.shuffle(items)
        sampled.extend(items[:per_cat])
    rng.shuffle(sampled)
    sampled = sampled[:n]

    # Build standardized format
    options_letters = list("ABCDEFGHIJ")
    result: list[dict] = []
    for i, item in enumerate(sampled):
        options = item.get("options", [])
        answer_idx = item.get("answer_index", 0)
        answer_letter = options_letters[answer_idx] if answer_idx < len(options_letters) else "A"
        choices = [f"{options_letters[j]}. {opt}" for j, opt in enumerate(options)]
        category = item.get("category", "other").lower()
        domain_cat = MMLU_CATEGORY_TO_DOMAIN.get(category, "factual")
        question_text = item.get("question", "")
        q_tok = _tok_approx(question_text + " ".join(options))
        result.append({
            "id": f"mmlu_pro_{i:05d}",
            "question": question_text,
            "answer": answer_letter,
            "choices": choices,
            "context": "",
            "category": domain_cat,
            "source_dataset": "mmlu_pro",
            "difficulty": "medium",
            "subject": category,
            "approx_tokens": q_tok,
        })

    _write_jsonl(out_dir / "test.jsonl", result)
    avg_tok = sum(r["approx_tokens"] for r in result) / max(1, len(result))
    stats = {
        "dataset": "mmlu_pro",
        "n_items": len(result),
        "avg_tokens": round(avg_tok, 1),
        "source_url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "categories": len(by_category),
    }
    print(f"  MMLU-Pro: {len(result):,} items, avg {avg_tok:.0f} tokens")
    return stats


def download_math500(out_dir: Path, n: int = 500, seed: int = 42) -> dict:
    """Download lighteval/MATH and sample 100 per difficulty level (1-5)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] datasets not installed.")
        return {}

    print("  Downloading MATH dataset …")
    # EleutherAI/hendrycks_math has per-subject configs, not a single 'all' config
    MATH_SUBJECTS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    ds = None
    dataset_name = None
    for name in ["EleutherAI/hendrycks_math", "lighteval/MATH"]:
        try:
            test = load_dataset(name, MATH_SUBJECTS[0], split="test")
            ds = None  # we'll iterate below
            dataset_name = name
            print(f"    Using {name} (per-subject)")
            break
        except Exception as e:
            print(f"  [WARN] {name}: {str(e)[:80]}")

    if dataset_name is None:
        print("  [ERROR] Could not load MATH dataset from any source")
        return {}

    # Load all subjects and combine
    all_rows: list = []
    for subj in MATH_SUBJECTS:
        try:
            subj_ds = load_dataset(dataset_name, subj, split="test")
            for item in subj_ds:
                item = dict(item)
                item["_subject"] = subj
                all_rows.append(item)
        except Exception as e:
            print(f"  [WARN] subject {subj}: {str(e)[:60]}")

    if not all_rows:
        print("  [ERROR] No MATH items loaded")
        return {}

    # Group by level
    by_level: dict[int, list] = defaultdict(list)
    for item in all_rows:
        lvl_raw = item.get("level", "Level 3")
        # "Level 1" → 1
        if isinstance(lvl_raw, str):
            match = re.search(r"(\d)", lvl_raw)
            lvl = int(match.group(1)) if match else 3
        else:
            lvl = int(lvl_raw)
        by_level[lvl].append(item)

    rng = random.Random(seed)
    per_level = n // 5
    sampled: list[dict] = []
    for lvl in range(1, 6):
        items = by_level.get(lvl, [])
        rng.shuffle(items)
        sampled.extend(items[:per_level])

    result: list[dict] = []
    for i, item in enumerate(sampled):
        solution = item.get("solution", "")
        # Extract boxed answer
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
        answer = boxed_match.group(1) if boxed_match else solution[-100:]
        lvl_raw = item.get("level", "Level 3")
        if isinstance(lvl_raw, str):
            match = re.search(r"(\d)", lvl_raw)
            lvl = int(match.group(1)) if match else 3
        else:
            lvl = int(lvl_raw)
        question = item.get("problem", item.get("question", ""))
        q_tok = _tok_approx(question)
        result.append({
            "id": f"math500_{i:05d}",
            "question": question,
            "answer": answer,
            "choices": None,
            "context": "",
            "category": "math",
            "source_dataset": "math500",
            "difficulty": DIFFICULTY_MAP.get(lvl, "medium"),
            "level": lvl,
            "subject": item.get("type", item.get("subject", "unknown")),
            "approx_tokens": q_tok,
        })

    _write_jsonl(out_dir / "test.jsonl", result)
    avg_tok = sum(r["approx_tokens"] for r in result) / max(1, len(result))
    stats = {
        "dataset": "math500",
        "n_items": len(result),
        "avg_tokens": round(avg_tok, 1),
        "source_url": "https://huggingface.co/datasets/lighteval/MATH",
    }
    print(f"  MATH-500: {len(result):,} items, avg {avg_tok:.0f} tokens")
    return stats


def download_livecodebench(out_dir: Path, n: int = 200, cutoff_date: str = "2024-07-01") -> dict:
    """Download livecodebench/code_generation_lite, take most recent n problems."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] datasets not installed.")
        return {}

    print("  Downloading livecodebench/code_generation_lite …")
    ds = None
    # Try multiple versions — LCB uses parquet in newer versions
    for name, kwargs in [
        ("livecodebench/code_generation_lite", {"split": "test"}),
        ("livecodebench/code_generation_lite", {"name": "release_v6", "split": "test"}),
        ("livecodebench/code_generation_lite", {"name": "release_v5", "split": "test"}),
        ("livecodebench/code_generation_lite", {"name": "release_v4", "split": "test"}),
        ("livecodebench/code_generation", {"split": "test"}),
    ]:
        try:
            ds = load_dataset(name, **kwargs)
            print(f"    Using {name} kwargs={kwargs}")
            break
        except Exception as e:
            print(f"  [WARN] {name} {kwargs}: {str(e)[:80]}")
    if ds is None:
        print("  [ERROR] Could not load LiveCodeBench from any source")
        return {}

    # All items are pre-2024-07-01 in test set, so take all and sort by date
    # Sort by date descending, take n most recent
    def sort_key(x: Any) -> str:
        return str(x.get("contest_date", x.get("date", "2024-01-01")))

    filtered = sorted(ds, key=sort_key, reverse=True)
    filtered = filtered[:n]

    result: list[dict] = []
    for i, item in enumerate(filtered):
        question = item.get("question_content", item.get("question", item.get("problem_statement", "")))
        tests = item.get("private_test_cases", item.get("public_test_cases", []))
        # Stringify tests for storage
        if isinstance(tests, list):
            tests_str = json.dumps(tests[:3])  # keep first 3 test cases
        else:
            tests_str = str(tests)[:500]
        q_tok = _tok_approx(question)
        difficulty_raw = item.get("difficulty", "medium")
        if isinstance(difficulty_raw, int):
            difficulty = DIFFICULTY_MAP.get(difficulty_raw, "medium")
        else:
            difficulty = str(difficulty_raw).lower() if difficulty_raw else "medium"
            if difficulty not in ("easy", "medium", "hard"):
                difficulty = "medium"
        result.append({
            "id": f"lcb_{i:05d}",
            "question": question,
            "answer": "pass@1",
            "choices": None,
            "context": "",
            "category": "code",
            "source_dataset": "livecodebench",
            "difficulty": difficulty,
            "test_cases": tests_str,
            "contest_date": str(item.get("contest_date", item.get("date", "2024-07-01"))),
            "approx_tokens": q_tok,
        })

    _write_jsonl(out_dir / "test.jsonl", result)
    avg_tok = sum(r["approx_tokens"] for r in result) / max(1, len(result))
    stats = {
        "dataset": "livecodebench",
        "n_items": len(result),
        "avg_tokens": round(avg_tok, 1),
        "source_url": "https://huggingface.co/datasets/livecodebench/code_generation_lite",
        "cutoff_date": cutoff_date,
    }
    print(f"  LiveCodeBench: {len(result):,} items, avg {avg_tok:.0f} tokens")
    return stats


def download_frames(out_dir: Path, n: int = 300) -> dict:
    """Download google/frames-benchmark, take first n questions from test split."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] datasets not installed.")
        return {}

    print("  Downloading google/frames-benchmark …")
    try:
        ds = load_dataset("google/frames-benchmark", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {}

    result: list[dict] = []
    for i, item in enumerate(list(ds)[:n]):
        question = item.get("Prompt", item.get("question", ""))
        answer = item.get("Answer", item.get("answer", ""))
        q_tok = _tok_approx(question + answer)
        result.append({
            "id": f"frames_{i:05d}",
            "question": question,
            "answer": answer,
            "choices": None,
            "context": "",
            "category": "rag",
            "source_dataset": "frames",
            "difficulty": "hard",  # FRAMES is a multi-hop, inherently hard
            "approx_tokens": q_tok,
        })

    _write_jsonl(out_dir / "test.jsonl", result)
    avg_tok = sum(r["approx_tokens"] for r in result) / max(1, len(result))
    stats = {
        "dataset": "frames",
        "n_items": len(result),
        "avg_tokens": round(avg_tok, 1),
        "source_url": "https://huggingface.co/datasets/google/frames-benchmark",
    }
    print(f"  FRAMES: {len(result):,} items, avg {avg_tok:.0f} tokens")
    return stats


def download_gaia(out_dir: Path, max_level: int = 2) -> dict:
    """Download gaia-benchmark/GAIA validation split, levels 1 and 2 only."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] datasets not installed.")
        return {}

    print("  Downloading gaia-benchmark/GAIA validation split …")
    ds = None
    for name, kwargs in [
        ("gaia-benchmark/GAIA", {"name": "2023_all", "split": "validation"}),
        ("gaia-benchmark/GAIA", {"split": "validation"}),
    ]:
        try:
            ds = load_dataset(name, **kwargs)
            print(f"    Using {name}")
            break
        except Exception as e:
            print(f"  [WARN] {name}: {str(e)[:80]}")
    if ds is None:
        print("  [ERROR] Could not load GAIA dataset")
        return {}

    result: list[dict] = []
    for i, item in enumerate(ds):
        level = item.get("Level", item.get("level", 3))
        try:
            level = int(level)
        except (ValueError, TypeError):
            level = 3
        if level > max_level:
            continue
        question = item.get("Question", item.get("question", ""))
        answer = item.get("Final answer", item.get("answer", ""))
        q_tok = _tok_approx(question + answer)
        difficulty = "easy" if level == 1 else "medium"
        result.append({
            "id": f"gaia_{i:05d}",
            "question": question,
            "answer": str(answer),
            "choices": None,
            "context": "",
            "category": "agent",
            "source_dataset": "gaia",
            "difficulty": difficulty,
            "level": level,
            "approx_tokens": q_tok,
        })

    _write_jsonl(out_dir / "test.jsonl", result)
    avg_tok = sum(r["approx_tokens"] for r in result) / max(1, len(result))
    stats = {
        "dataset": "gaia",
        "n_items": len(result),
        "avg_tokens": round(avg_tok, 1),
        "source_url": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        "levels_included": list(range(1, max_level + 1)),
    }
    print(f"  GAIA: {len(result):,} items (levels 1-{max_level}), avg {avg_tok:.0f} tokens")
    return stats


def download_sharegpt(out_dir: Path, n: int = 500, min_tokens: int = 1000, seed: int = 42) -> dict:
    """Download ShareGPT conversations with ≥ min_tokens approximate length."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] datasets not installed.")
        return {}

    print("  Downloading anon8231489123/ShareGPT_Vicuna_unfiltered …")
    ds = None
    for name, kwargs in [
        ("anon8231489123/ShareGPT_Vicuna_unfiltered",
         {"data_files": "ShareGPT_V3_unfiltered_cleaned_split.json", "split": "train"}),
        ("anon8231489123/ShareGPT_Vicuna_unfiltered", {"split": "train"}),
        ("RyokoAI/ShareGPT52K", {"split": "train"}),
        ("liyucheng/ShareGPT90K", {"split": "train"}),
    ]:
        try:
            ds = load_dataset(name, **kwargs)
            print(f"    Using {name}")
            break
        except Exception as e:
            print(f"  [WARN] {name}: {str(e)[:80]}")
    if ds is None:
        print("  [ERROR] Could not load ShareGPT dataset")
        return {}

    rng = random.Random(seed)
    all_items = list(ds)
    rng.shuffle(all_items)

    result: list[dict] = []
    for item in all_items:
        if len(result) >= n:
            break
        convs = item.get("conversations", item.get("conversation", []))
        if not convs:
            continue
        # Compute total text length
        total_text = " ".join(
            c.get("value", "") for c in convs if isinstance(c, dict)
        )
        if _tok_approx(total_text) < min_tokens:
            continue
        # Extract final question (last human turn)
        human_turns = [c for c in convs if isinstance(c, dict) and c.get("from", "") in ("human", "user")]
        final_q = human_turns[-1].get("value", "") if human_turns else ""
        q_tok = _tok_approx(total_text)
        result.append({
            "id": item.get("id", f"sharegpt_{len(result):05d}"),
            "question": final_q,
            "answer": "",
            "choices": None,
            "context": "",
            "category": "factual",
            "source_dataset": "sharegpt",
            "difficulty": "medium",
            "conversations": convs,
            "approx_tokens": q_tok,
        })

    _write_jsonl(out_dir / "test.jsonl", result)
    avg_tok = sum(r["approx_tokens"] for r in result) / max(1, len(result))
    stats = {
        "dataset": "sharegpt",
        "n_items": len(result),
        "avg_tokens": round(avg_tok, 1),
        "source_url": "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered",
        "min_tokens_filter": min_tokens,
    }
    print(f"  ShareGPT: {len(result):,} items (≥{min_tokens} tokens), avg {avg_tok:.0f} tokens")
    return stats


def validate_all(data_dir: Path) -> None:
    """Validate all downloaded datasets."""
    print("\n=== Dataset Validation ===")
    datasets = ["mmlu_pro", "math500", "livecodebench", "frames", "gaia", "sharegpt"]
    all_ok = True
    for name in datasets:
        path = data_dir / name / "test.jsonl"
        if not path.exists():
            print(f"  ✗ {name}: MISSING ({path})")
            all_ok = False
            continue
        count = 0
        errors = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Check required fields
                    for field in ("id", "question", "answer", "category", "source_dataset", "difficulty"):
                        if field not in item:
                            errors += 1
                    count += 1
                except json.JSONDecodeError:
                    errors += 1
        status = "✓" if errors == 0 else f"✗ ({errors} errors)"
        print(f"  {status} {name}: {count:,} items")
    if all_ok:
        print("\nAll datasets OK")
    # Print stats if available
    stats_path = data_dir / "download_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print("\n=== Download Stats ===")
        for name, s in stats.items():
            if isinstance(s, dict):
                print(f"  {name}: {s.get('n_items', '?'):,} items, avg {s.get('avg_tokens', '?')} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Phase 2 benchmark datasets.")
    parser.add_argument("--dataset", default="all", help="Dataset to download (default: all)")
    parser.add_argument("--validate", action="store_true", help="Validate existing downloads")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(__file__).parent

    if args.validate:
        validate_all(data_dir)
        return

    all_stats: dict[str, dict] = {}

    datasets = {
        "mmlu_pro": lambda: download_mmlu_pro(data_dir / "mmlu_pro", seed=args.seed),
        "math500": lambda: download_math500(data_dir / "math500", seed=args.seed),
        "livecodebench": lambda: download_livecodebench(data_dir / "livecodebench"),
        "frames": lambda: download_frames(data_dir / "frames"),
        "gaia": lambda: download_gaia(data_dir / "gaia"),
        "sharegpt": lambda: download_sharegpt(data_dir / "sharegpt", seed=args.seed),
    }

    if args.dataset == "all":
        targets = list(datasets.keys())
    else:
        targets = [args.dataset]

    for name in targets:
        if name not in datasets:
            print(f"Unknown dataset: {name}. Choose from: {', '.join(datasets)}")
            continue
        print(f"\n[{name}]")
        t0 = time.time()
        try:
            stats = datasets[name]()
            if stats:
                all_stats[name] = stats
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")
        print(f"  Done in {time.time() - t0:.1f}s")

    # Save stats
    if all_stats:
        with open(STATS_FILE, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nStats saved → {STATS_FILE}")

    print("\n=== Summary ===")
    for name, s in all_stats.items():
        print(f"  {name}: {s.get('n_items', 0):,} items, avg {s.get('avg_tokens', 0):.0f} tokens")


if __name__ == "__main__":
    main()
