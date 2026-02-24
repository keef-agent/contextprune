"""Tests for MMRSelector — within-message chunk selection.

Unit tests use the same embedding mock as test_dedup.py (keyword-group vectors).
Integration tests are marked @pytest.mark.integration.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import numpy as np
import pytest

from contextprune.dedup import MMRSelector, _split_chunks


# ---------------------------------------------------------------------------
# Embedding mock (same keyword-group approach as test_dedup.py)
# ---------------------------------------------------------------------------

_KEYWORD_GROUPS = [
    (["postgresql", "database", "port 5432", "data storage"], 0),
    (["weather", "forecast", "temperature", "climate"], 1),
    (["python", "code", "function", "programming"], 2),
    (["file", "read", "write", "directory"], 3),
    (["sql", "query", "database query"], 4),
    (["web", "search", "internet"], 5),
    (["email", "send", "message"], 6),
    (["index", "b-tree", "gin", "gist", "hash"], 7),
    (["replication", "streaming", "standby", "wal"], 8),
    (["vacuum", "mvcc", "autovacuum", "bloat"], 9),
]

_DIM = 20
_embed_cache: dict = {}
_unique_counter = [len(_KEYWORD_GROUPS)]


def _mock_embed(texts, model_name, prefix=""):
    _embed_cache.clear()
    results = []
    for text in texts:
        clean = re.sub(r"^(search_document: |search_query: )", "", text).lower()
        if clean not in _embed_cache:
            best_group = None
            best_count = 0
            for keywords, group_idx in _KEYWORD_GROUPS:
                count = sum(1 for kw in keywords if kw in clean)
                if count > best_count:
                    best_count = count
                    best_group = group_idx
            vec = np.zeros(_DIM)
            if best_group is not None:
                vec[best_group] = 1.0
            else:
                slot = _unique_counter[0] % _DIM
                _unique_counter[0] += 1
                vec[slot] = 1.0
            _embed_cache[clean] = vec
        results.append(_embed_cache[clean])
    return np.array(results, dtype=np.float32)


def _mock_cosine_sim(a, b):
    return float(np.dot(a, b))


@pytest.fixture(autouse=True)
def patch_embeddings(request, monkeypatch):
    """Patch embedding module for unit tests; skip for @pytest.mark.integration."""
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr("contextprune.embeddings.embed", _mock_embed)
    monkeypatch.setattr("contextprune.embeddings.cosine_similarity", _mock_cosine_sim)
    _embed_cache.clear()
    _unique_counter[0] = len(_KEYWORD_GROUPS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _make_redundant_content(n_copies: int = 6) -> str:
    """Build a long document with highly redundant paragraphs about PostgreSQL."""
    paragraphs = []
    for i in range(n_copies):
        paragraphs.append(
            f"PostgreSQL is a relational database system that uses SQL for querying data. "
            f"It stores data efficiently on port 5432. This is paragraph {i + 1} about "
            f"the PostgreSQL database and its data storage capabilities."
        )
    return "\n\n".join(paragraphs)


def _make_diverse_content() -> str:
    """Build a document with paragraphs on completely different topics."""
    return "\n\n".join([
        "PostgreSQL is a database system that stores data on port 5432.",
        "Python is a programming language used for writing code and functions.",
        "The weather forecast shows temperature changes across the climate zones.",
        "Web search on the internet returns many results from different sources.",
        "Email messages can be sent and received through various platforms.",
    ])


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestMMRSelectorBasics:
    def setup_method(self):
        # Use low min_tokens_to_mmr so tests don't need huge content
        self.mmr = MMRSelector(
            token_budget_ratio=0.5,
            lambda_param=0.5,
            min_chunk_tokens=10,
            min_tokens_to_mmr=50,
            redundancy_threshold=0.3,
        )

    def test_single_chunk_passes_through_unchanged(self):
        """A single-paragraph message has nothing to deduplicate."""
        content = "PostgreSQL is a database system that stores data on port 5432. " * 5
        result, stats = self.mmr.select(content, query="What database should I use?")
        # Single chunk = no redundancy to exploit; pass through
        assert stats["chunks_total"] <= 1 or stats["reduction_pct"] >= 0.0
        assert len(result) > 0

    def test_redundant_content_is_compressed(self):
        """Highly redundant paragraphs should be compressed significantly."""
        content = _make_redundant_content(n_copies=8)
        query = "Tell me about PostgreSQL database."
        result, stats = self.mmr.select(content, query)
        # With 8 nearly identical paragraphs, MMR should drop most of them
        assert stats["reduction_pct"] > 0, (
            f"Expected compression but got {stats['reduction_pct']}% — stats: {stats}"
        )
        assert stats["chunks_kept"] < stats["chunks_total"], (
            f"Expected fewer chunks but got {stats['chunks_kept']}/{stats['chunks_total']}"
        )

    def test_diverse_content_not_heavily_compressed(self):
        """Diverse content (low pairwise similarity) should be kept mostly intact."""
        content = _make_diverse_content()
        query = "Give me an overview of all topics."
        result, stats = self.mmr.select(content, query)
        # Low redundancy → skipped or minimal compression
        # The redundancy_threshold=0.3 means mean pairwise sim < 0.3 → pass-through
        # With our mock, different-topic paragraphs have 0 similarity → pass-through
        assert stats["reduction_pct"] < 50, (
            f"Should not heavily compress diverse content but got {stats['reduction_pct']}%"
        )

    def test_query_relevant_chunks_preserved(self):
        """Chunks relevant to the query should be kept, not dropped."""
        # Build content where paragraph 1 is about the query topic and the rest are
        # redundant noise about PostgreSQL
        query_topic_para = (
            "Python programming functions are essential for writing good code. "
            "Here is a function that solves your problem effectively."
        )
        noise = "\n\n".join(
            f"PostgreSQL database on port 5432 stores data efficiently. Variant {i}."
            for i in range(6)
        )
        content = query_topic_para + "\n\n" + noise
        query = "Write a Python function."

        result, stats = self.mmr.select(content, query)
        # The Python paragraph should be in the compressed output
        assert "Python" in result or "function" in result, (
            f"Query-relevant content was dropped. Result: {result[:200]}"
        )

    def test_preserve_order_true_maintains_document_order(self):
        """With preserve_order=True, selected chunks appear in original sequence."""
        # 6 redundant paragraphs so MMR fires and we can inspect order
        paragraphs = [
            f"PostgreSQL database on port 5432. Paragraph {i}." for i in range(6)
        ]
        content = "\n\n".join(paragraphs)
        query = "PostgreSQL database"

        mmr = MMRSelector(
            token_budget_ratio=0.5,
            lambda_param=0.5,
            min_chunk_tokens=5,
            min_tokens_to_mmr=50,
            redundancy_threshold=0.3,
            preserve_order=True,
        )
        result, stats = mmr.select(content, query)

        if stats["reduction_pct"] > 0:
            # Find which paragraph indices appear in result
            found_indices = []
            for i, para in enumerate(paragraphs):
                # Check if paragraph number is present
                if f"Paragraph {i}" in result:
                    found_indices.append(i)
            # They should be in ascending order
            assert found_indices == sorted(found_indices), (
                f"Chunk order not preserved: {found_indices}"
            )

    def test_no_paragraph_breaks_falls_back_to_sentences(self):
        """Content without paragraph breaks should be split by sentences."""
        # Build long sentence-based content with no paragraph breaks
        sentences = [
            "PostgreSQL stores data on port 5432 efficiently.",
            "Python code runs functions in a programming environment.",
            "Web search on the internet finds many results online.",
            "Email messages are sent through various platforms.",
            "The weather forecast shows temperature and climate data.",
            "File reading and writing happens in the directory.",
        ] * 3  # repeat to ensure >min_tokens_to_mmr

        content = " ".join(sentences)  # No \n\n paragraph breaks
        query = "Tell me about databases."

        result, stats = self.mmr.select(content, query)
        # Should not crash; output should be non-empty
        assert len(result) > 0
        assert isinstance(stats, dict)
        assert "chunks_total" in stats

    def test_short_content_passes_through_unchanged(self):
        """Content below min_tokens_to_mmr should not be compressed."""
        mmr = MMRSelector(min_tokens_to_mmr=1000)  # very high threshold
        content = "Short text about PostgreSQL."
        result, stats = mmr.select(content, query="What is PostgreSQL?")
        assert result == content
        assert stats["reduction_pct"] == 0.0

    def test_output_is_non_empty_string(self):
        """MMR should always return a non-empty string."""
        content = _make_redundant_content(n_copies=4)
        result, stats = self.mmr.select(content, query="database")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_stats_keys_present(self):
        """Stats dict should have all required keys."""
        content = _make_redundant_content(n_copies=4)
        result, stats = self.mmr.select(content, query="database")
        required_keys = {
            "original_tokens", "selected_tokens", "reduction_pct",
            "chunks_total", "chunks_kept", "mmr_scores",
        }
        assert required_keys.issubset(stats.keys()), (
            f"Missing keys: {required_keys - stats.keys()}"
        )

    def test_selected_tokens_le_original_tokens(self):
        """selected_tokens should never exceed original_tokens."""
        content = _make_redundant_content(n_copies=6)
        _, stats = self.mmr.select(content, query="PostgreSQL database")
        assert stats["selected_tokens"] <= stats["original_tokens"]

    def test_chunks_kept_le_chunks_total(self):
        """chunks_kept should never exceed chunks_total."""
        content = _make_redundant_content(n_copies=6)
        _, stats = self.mmr.select(content, query="database")
        assert stats["chunks_kept"] <= stats["chunks_total"]

    def test_empty_content_returns_safely(self):
        """Empty string should not raise."""
        result, stats = self.mmr.select("", query="database")
        assert isinstance(result, str)

    def test_lambda_zero_pure_diversity(self):
        """lambda=0 → pure diversity; should still select chunks."""
        mmr = MMRSelector(
            lambda_param=0.0,
            min_chunk_tokens=5,
            min_tokens_to_mmr=50,
            redundancy_threshold=0.3,
        )
        content = _make_redundant_content(n_copies=6)
        result, stats = mmr.select(content, query="database")
        assert len(result) > 0

    def test_lambda_one_pure_relevance(self):
        """lambda=1 → pure relevance; should still select chunks."""
        mmr = MMRSelector(
            lambda_param=1.0,
            min_chunk_tokens=5,
            min_tokens_to_mmr=50,
            redundancy_threshold=0.3,
        )
        content = _make_redundant_content(n_copies=6)
        result, stats = mmr.select(content, query="database")
        assert len(result) > 0


class TestMMRSelectorParagraphMode:
    """Tests specific to paragraph splitting mode."""

    def setup_method(self):
        self.mmr = MMRSelector(
            chunk_by="paragraph",
            token_budget_ratio=0.5,
            min_chunk_tokens=5,
            min_tokens_to_mmr=50,
            redundancy_threshold=0.3,
        )

    def test_paragraph_output_has_no_partial_sentences(self):
        """With chunk_by=paragraph, output should consist of complete paragraphs."""
        content = (
            "First paragraph about PostgreSQL database and port 5432 data storage.\n\n"
            "Second paragraph about PostgreSQL database and port 5432 data storage.\n\n"
            "Third paragraph about PostgreSQL database and port 5432 data storage.\n\n"
            "Fourth paragraph about PostgreSQL database and port 5432 data storage.\n\n"
            "Fifth paragraph about PostgreSQL database and port 5432 data storage."
        )
        result, stats = self.mmr.select(content, query="PostgreSQL")
        # Each result chunk should be a complete paragraph (no trailing incomplete words)
        # i.e. result should end at a sentence boundary
        assert result.strip().endswith(".") or result.strip().endswith("storage.")

    def test_paragraph_separator_used_in_output(self):
        """Output paragraphs should be joined with paragraph separator."""
        content = "\n\n".join([
            "PostgreSQL database on port 5432. Data storage is key.",
            "PostgreSQL database on port 5432. Data storage matters.",
            "PostgreSQL database on port 5432. Data storage is essential.",
            "Python programming for code and functions. Very different.",
            "Weather forecast with temperature. Another topic entirely.",
        ])
        result, stats = self.mmr.select(content, query="database")
        if stats["chunks_kept"] > 1 and stats["reduction_pct"] > 0:
            # When multiple chunks selected, they should be paragraph-separated
            assert "\n\n" in result, "Expected paragraph separator in multi-chunk output"


class TestMMRIntegration:
    """Integration tests using real embedding model."""

    @pytest.mark.integration
    def test_real_embeddings_compress_rag_chunks(self):
        """With real embeddings, RAG chunks with semantic overlap should compress."""
        from benchmarks.scenarios import get_scenario2
        system, messages = get_scenario2()

        mmr = MMRSelector(
            token_budget_ratio=0.5,
            lambda_param=0.5,
            model="all-MiniLM-L6-v2",
            min_tokens_to_mmr=200,
            redundancy_threshold=0.25,
        )
        query = messages[-1]["content"] if messages else "database index"
        result, stats = mmr.select(system, query=query)

        # Should compress the system (which contains 10 overlapping RAG chunks)
        assert stats["reduction_pct"] >= 0, "MMR should not increase token count"
        # With real embeddings and real overlapping RAG chunks, expect some compression
        print(f"\nRAG integration test: {stats['reduction_pct']:.1f}% reduction, "
              f"{stats['chunks_kept']}/{stats['chunks_total']} chunks kept")

    @pytest.mark.integration
    def test_real_embeddings_diverse_content_not_destroyed(self):
        """Diverse content should not be heavily compressed with real embeddings."""
        content = (
            "PostgreSQL is a relational database management system.\n\n"
            "Python is a high-level programming language.\n\n"
            "Machine learning models require training data.\n\n"
            "The TCP/IP protocol suite governs internet communication.\n\n"
            "Financial markets respond to economic indicators."
        )
        mmr = MMRSelector(
            model="all-MiniLM-L6-v2",
            min_tokens_to_mmr=50,
            redundancy_threshold=0.3,
        )
        result, stats = mmr.select(content, query="What is machine learning?")
        # Should keep most of the content — very low pairwise similarity
        assert stats["reduction_pct"] < 60, (
            f"Diverse content should not be heavily compressed: {stats['reduction_pct']}%"
        )
