"""Semantic deduplication across messages.

Detects and removes semantically redundant content across messages using
TF-IDF cosine similarity. Sentences that are too similar to earlier sentences
get pruned.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


def _sentence_split(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on sentence-ending punctuation followed by whitespace or end of string
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Build IDF scores from a list of tokenized documents."""
    n = len(documents)
    if n == 0:
        return {}
    df: Counter = Counter()
    for doc in documents:
        unique = set(doc)
        for token in unique:
            df[token] += 1
    return {token: math.log((n + 1) / (freq + 1)) + 1 for token, freq in df.items()}


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute TF-IDF vector for a tokenized document."""
    tf: Counter = Counter(tokens)
    vec: Dict[str, float] = {}
    for token, count in tf.items():
        vec[token] = count * idf.get(token, 1.0)
    return vec


def _cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticDeduplicator:
    """Remove semantically redundant sentences across messages."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold

    def deduplicate(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
        """Deduplicate messages and system prompt.

        Returns (new_messages, new_system, sentences_removed).
        Later messages are pruned in favor of earlier ones.
        """
        # Collect all sentences with their source info
        all_sentences: List[Tuple[str, str, int, int]] = []
        # (sentence_text, source_type, source_index, sentence_index)

        if system:
            for i, sent in enumerate(_sentence_split(system)):
                all_sentences.append((sent, "system", 0, i))

        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                for i, sent in enumerate(_sentence_split(content)):
                    all_sentences.append((sent, "message", msg_idx, i))

        if len(all_sentences) < 2:
            return messages, system, 0

        # Tokenize all sentences
        tokenized = [_tokenize(s[0]) for s in all_sentences]

        # Build IDF across all sentences
        idf = _build_idf(tokenized)

        # Compute TF-IDF vectors
        vectors = [_tfidf_vector(t, idf) for t in tokenized]

        # Mark duplicates: later sentences that are too similar to earlier ones
        removed_indices: Set[int] = set()
        for i in range(len(all_sentences)):
            if i in removed_indices:
                continue
            # Skip very short sentences (less than 3 tokens) -- not worth deduping
            if len(tokenized[i]) < 3:
                continue
            for j in range(i + 1, len(all_sentences)):
                if j in removed_indices:
                    continue
                if len(tokenized[j]) < 3:
                    continue
                sim = _cosine_sim(vectors[i], vectors[j])
                if sim >= self.similarity_threshold:
                    removed_indices.add(j)

        removed_count = len(removed_indices)

        # Rebuild system prompt
        new_system = system
        if system:
            system_sentences = [
                all_sentences[i][0]
                for i in range(len(all_sentences))
                if all_sentences[i][1] == "system" and i not in removed_indices
            ]
            if system_sentences:
                new_system = " ".join(system_sentences)
            elif removed_count > 0:
                # All system sentences removed -- keep the original
                new_system = system

        # Rebuild messages
        new_messages = []
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                kept_sentences = [
                    all_sentences[i][0]
                    for i in range(len(all_sentences))
                    if all_sentences[i][1] == "message"
                    and all_sentences[i][2] == msg_idx
                    and i not in removed_indices
                ]
                if kept_sentences:
                    new_msg = dict(msg)
                    new_msg["content"] = " ".join(kept_sentences)
                    new_messages.append(new_msg)
                else:
                    # Keep the message but with minimal content to preserve structure
                    new_msg = dict(msg)
                    new_msg["content"] = content
                    new_messages.append(new_msg)
            else:
                # Non-string content (tool results, images, etc) -- pass through
                new_messages.append(msg)

        return new_messages, new_system, removed_count
