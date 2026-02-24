"""
Experiment 2: Semantic Preservation

For each compressed message from Exp 1, computes cosine similarity between
the original and compressed text using sentence-transformers all-MiniLM-L6-v2.

Goal: verify that deduplication removes redundancy without losing meaning.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/home/keith/contextprune")

import numpy as np
from contextprune.dedup import SemanticDeduplicator
from benchmarks.scenarios import get_all_scenarios


def _get_encoder():
    """Load sentence-transformers model (cached after first call)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def collect_text_pairs(
    system: str,
    messages: List[Dict[str, Any]],
    new_system: str,
    new_messages: List[Dict[str, Any]],
) -> List[Tuple[str, str, str]]:
    """
    Collect (original_text, compressed_text, label) pairs.
    Only includes items where content actually changed.
    """
    pairs = []

    # System prompt
    if system and new_system and system != new_system and len(new_system.strip()) > 10:
        pairs.append((system, new_system, "system_prompt"))

    # Messages
    for i, (orig_msg, new_msg) in enumerate(zip(messages, new_messages)):
        orig_content = orig_msg.get("content", "")
        new_content = new_msg.get("content", "")
        if (
            isinstance(orig_content, str)
            and isinstance(new_content, str)
            and orig_content != new_content
            and len(new_content.strip()) > 10
        ):
            role = orig_msg.get("role", "msg")
            pairs.append((orig_content, new_content, f"{role}[{i}]"))

    return pairs


def run_exp2() -> List[Dict[str, Any]]:
    """Run Experiment 2 — semantic preservation."""
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = _get_encoder()
    print("Model loaded.")

    dedup = SemanticDeduplicator(similarity_threshold=0.85)
    scenarios = get_all_scenarios()
    results = []

    for name, system, messages, tools in scenarios:
        new_messages, new_system, removed = dedup.deduplicate(messages, system=system)

        pairs = collect_text_pairs(system, messages, new_system or system, new_messages)

        if not pairs:
            # No content changed — compression was trivial or nothing to dedup
            results.append({
                "scenario": name,
                "pairs_compared": 0,
                "mean_similarity": 1.0,
                "min_similarity": 1.0,
                "max_similarity": 1.0,
                "pct_above_085": 100.0,
                "sentences_removed": removed,
            })
            continue

        # Embed all originals and compressed texts in batch
        originals = [p[0] for p in pairs]
        compressed = [p[1] for p in pairs]

        orig_embs = model.encode(originals, convert_to_numpy=True, show_progress_bar=False)
        comp_embs = model.encode(compressed, convert_to_numpy=True, show_progress_bar=False)

        sims = [cosine_sim(orig_embs[i], comp_embs[i]) for i in range(len(pairs))]

        results.append({
            "scenario": name,
            "pairs_compared": len(pairs),
            "mean_similarity": round(float(np.mean(sims)), 4),
            "min_similarity": round(float(np.min(sims)), 4),
            "max_similarity": round(float(np.max(sims)), 4),
            "pct_above_085": round(100.0 * sum(1 for s in sims if s >= 0.85) / len(sims), 1),
            "sentences_removed": removed,
            "per_pair": [
                {"label": pairs[i][2], "similarity": round(sims[i], 4)}
                for i in range(len(pairs))
            ],
        })

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Semantic Preservation")
    print("=" * 80)
    print(f"\n{'Scenario':<22} {'Pairs':>6} {'Mean Sim':>10} {'Min':>8} {'Max':>8} {'>0.85':>8} {'SentsRm':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['scenario']:<22} {r['pairs_compared']:>6} {r['mean_similarity']:>10.4f} "
            f"{r['min_similarity']:>8.4f} {r['max_similarity']:>8.4f} "
            f"{r['pct_above_085']:>7.1f}% {r['sentences_removed']:>8}"
        )


if __name__ == "__main__":
    results = run_exp2()
    print_results(results)
