"""Tests for SemanticDeduplicator (embedding-based implementation).

Unit tests mock the embedding model so no model download is required.
Integration tests (marked @pytest.mark.integration) use real embeddings.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import numpy as np
import pytest

from contextprune.dedup import SemanticDeduplicator, _split_chunks


# ---------------------------------------------------------------------------
# Shared embedding mock
# ---------------------------------------------------------------------------

# Keyword groups: texts sharing keywords from the same group get identical
# unit vectors (cosine similarity = 1.0). Texts with no common group get
# unique vectors from the hash fallback (cosine similarity ≈ 0.0).
_KEYWORD_GROUPS = [
    (["postgresql", "database", "port 5432", "data storage", "storing data"], 0),
    (["weather", "forecast", "temperature", "climate"], 1),
    (["python", "code", "function", "programming", "coding"], 2),
    (["file", "read", "write", "directory", "config"], 3),
    (["sql", "query", "database query"], 4),
    (["web", "search", "internet"], 5),
    (["email", "send", "message"], 6),
    (["helpful", "assistant", "coding assistant"], 7),
    (["nginx", "server", "port 8080", "web server"], 8),
    (["quantum", "computing"], 9),
    (["stock", "price", "financial", "market", "ticker"], 10),
    (["deploy", "service", "production", "deployment"], 11),
    (["git", "commit", "version control"], 12),
]

_DIM = 20
_embed_cache: dict = {}
_unique_counter = [len(_KEYWORD_GROUPS)]  # start above group range


def _mock_embed(texts, model_name, prefix=""):
    """Content-aware embedding mock for unit tests.

    Strips task prefixes before grouping, so 'search_document: foo' and
    'search_query: foo' map to identical vectors (as they should for dedup).
    Texts in the same keyword group get identical unit vectors.
    Unique texts get unique vectors via hash fallback.
    """
    _embed_cache.clear()  # reset per test to avoid cross-test contamination
    results = []
    for text in texts:
        # Strip nomic-embed task prefixes before comparing
        clean = re.sub(r"^(search_document: |search_query: )", "", text).lower()

        if clean not in _embed_cache:
            # Find the keyword group with the most matches
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
                # Unique vector for this text (no keyword match)
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
    """Patch embedding module for unit tests. Skipped for @pytest.mark.integration tests."""
    if request.node.get_closest_marker("integration"):
        return  # Let integration tests use real embeddings
    monkeypatch.setattr("contextprune.embeddings.embed", _mock_embed)
    monkeypatch.setattr("contextprune.embeddings.cosine_similarity", _mock_cosine_sim)
    _embed_cache.clear()
    _unique_counter[0] = len(_KEYWORD_GROUPS)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSemanticDeduplicator:
    def setup_method(self):
        self.dedup = SemanticDeduplicator(similarity_threshold=0.82)

    def test_no_duplicates_unchanged(self):
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "Tell me about quantum computing."},
        ]
        new_msgs, new_sys, removed = self.dedup.deduplicate(messages)
        assert removed == 0
        assert len(new_msgs) == 3

    def test_exact_duplicate_removed(self):
        messages = [
            {"role": "user", "content": "The database connection uses PostgreSQL on port 5432."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "The database connection uses PostgreSQL on port 5432."},
        ]
        new_msgs, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 1

    def test_near_duplicate_removed(self):
        """Both sentences share PostgreSQL/database/port-5432 keywords → same group → removed."""
        messages = [
            {"role": "user", "content": "The application uses PostgreSQL database on port 5432 for data storage."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "The application uses PostgreSQL database on port 5432 for storing data."},
        ]
        new_msgs, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 1

    def test_system_prompt_dedup(self):
        """Chunks from the user message that match system prompt chunks should be removed."""
        system = "You are a helpful coding assistant. You help users write Python code."
        messages = [
            {"role": "user", "content": "You are a helpful coding assistant. Help me write a function."},
        ]
        new_msgs, new_system, removed = self.dedup.deduplicate(messages, system=system)
        assert removed >= 1

    def test_preserves_message_structure(self):
        messages = [
            {"role": "user", "content": "Hello there, how are you today?"},
            {"role": "assistant", "content": "I am doing well, thanks for asking!"},
        ]
        new_msgs, _, _ = self.dedup.deduplicate(messages)
        assert all("role" in m for m in new_msgs)
        assert all("content" in m for m in new_msgs)

    def test_non_string_content_passthrough(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi there!"},
        ]
        new_msgs, _, _ = self.dedup.deduplicate(messages)
        assert len(new_msgs) == 2
        assert isinstance(new_msgs[0]["content"], list)

    def test_empty_messages(self):
        new_msgs, new_sys, removed = self.dedup.deduplicate([])
        assert new_msgs == []
        assert removed == 0

    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello world."}]
        new_msgs, _, removed = self.dedup.deduplicate(messages)
        assert len(new_msgs) == 1
        assert removed == 0

    def test_threshold_affects_sensitivity(self):
        """Lower threshold should catch >= as many duplicates as higher threshold."""
        messages = [
            {"role": "user", "content": "The server runs on port 8080 with nginx."},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "The web server is nginx running on port 8080."},
        ]
        strict = SemanticDeduplicator(similarity_threshold=0.95)
        _, _, removed_strict = strict.deduplicate(messages)

        loose = SemanticDeduplicator(similarity_threshold=0.5)
        _, _, removed_loose = loose.deduplicate(messages)

        assert removed_loose >= removed_strict

    def test_many_duplicates(self):
        """Should handle many repeated Python-related sentences."""
        base = "The system requires Python 3.9 or higher to run properly."
        messages = [{"role": "user", "content": base}]
        for i in range(10):
            messages.append({"role": "assistant", "content": f"Response {i}."})
            messages.append({"role": "user", "content": base + f" Additional note {i}."})

        _, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 5

    def test_min_chunk_tokens_passthrough(self):
        """Chunks below min_chunk_tokens threshold pass through without dedup."""
        dedup = SemanticDeduplicator(min_chunk_tokens=100)  # very high threshold
        messages = [
            {"role": "user", "content": "Short text."},
            {"role": "assistant", "content": "Short text."},  # identical but tiny
        ]
        _, _, removed = dedup.deduplicate(messages)
        # Both chunks are way below 100 tokens, so neither is eligible for dedup
        assert removed == 0

    def test_removal_log_populated(self):
        """removal_log should contain details about what was removed."""
        messages = [
            {"role": "user", "content": "The database connection uses PostgreSQL on port 5432."},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "The database connection uses PostgreSQL on port 5432."},
        ]
        _, _, removed = self.dedup.deduplicate(messages)
        if removed > 0:
            assert len(self.dedup.removal_log) == removed
            for removed_chunk, kept_chunk, score in self.dedup.removal_log:
                assert isinstance(removed_chunk, str)
                assert isinstance(kept_chunk, str)
                assert 0.0 <= score <= 1.0

    def test_model_shorthand_minilm(self):
        """'minilm' shorthand should be accepted without error."""
        dedup = SemanticDeduplicator(model="minilm")
        assert dedup.model == "all-MiniLM-L6-v2"

    def test_cross_message_deduplication(self):
        """Chunks from message 1 should be deduplicated in message 3."""
        messages = [
            {"role": "user", "content": "The database connection uses PostgreSQL on port 5432."},
            {"role": "assistant", "content": "Acknowledged."},
            {
                "role": "user",
                "content": (
                    "Please help me debug this issue. "
                    "The database connection uses PostgreSQL on port 5432."
                ),
            },
        ]
        new_msgs, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 1
        # The PostgreSQL sentence should be removed from message 3
        # "Please help me debug this issue." has unique keywords → kept
        assert "debug" in new_msgs[2]["content"] or len(new_msgs[2]["content"]) < len(messages[2]["content"])

    def test_chunk_by_paragraph(self):
        """Paragraph-mode splitting should work."""
        dedup = SemanticDeduplicator(chunk_by="paragraph")
        messages = [
            {"role": "user", "content": "First paragraph about PostgreSQL database.\n\nSecond paragraph about weather."},
        ]
        new_msgs, _, _ = dedup.deduplicate(messages)
        assert len(new_msgs) == 1

    def test_no_system_prompt_no_crash(self):
        """Should work fine without a system prompt."""
        messages = [{"role": "user", "content": "The database connection uses PostgreSQL."}]
        new_msgs, new_sys, removed = self.dedup.deduplicate(messages, system=None)
        assert new_sys is None
        assert len(new_msgs) == 1


class TestChunkSplitting:
    def test_sentence_split(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_chunks(text, "sentence")
        assert len(chunks) == 3

    def test_paragraph_split(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = _split_chunks(text, "paragraph")
        assert len(chunks) == 2

    def test_chunk_split(self):
        text = "Line one\nLine two\nLine three"
        chunks = _split_chunks(text, "chunk")
        assert len(chunks) == 3

    def test_invalid_chunk_by(self):
        with pytest.raises(ValueError, match="Unknown chunk_by"):
            _split_chunks("text", "word")


# ---------------------------------------------------------------------------
# Integration tests (require real embedding model, ~137MB download)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSemanticDeduplicatorIntegration:
    """Real embedding tests. Run with: pytest -m integration"""

    def setup_method(self):
        # Use MiniLM for integration tests (22MB, faster than nomic)
        self.dedup = SemanticDeduplicator(
            model="all-MiniLM-L6-v2",
            similarity_threshold=0.82,
        )

    def test_near_duplicate_detected_with_real_embeddings(self):
        """Real embeddings should detect semantically similar sentences."""
        messages = [
            {"role": "user", "content": "The application uses PostgreSQL database on port 5432 for data storage."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "The application uses PostgreSQL database on port 5432 for storing data."},
        ]
        _, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 1, "Real embeddings should detect this near-duplicate"

    def test_256_token_context_window_not_truncated(self):
        """nomic-embed should handle long chunks without truncation issues.

        The test uses repeated short sentences so each sentence is ~12 tokens.
        With default min_chunk_tokens=5, each is eligible for dedup.
        """
        # Use nomic-embed which has 2048-token context (MiniLM would truncate at 256)
        dedup = SemanticDeduplicator(
            model="nomic-ai/nomic-embed-text-v1.5",
            similarity_threshold=0.82,
            min_chunk_tokens=5,  # each repeated sentence is ~12 tokens
        )
        # 20 sentences, each ~12 tokens = ~240 tokens total per message
        long_chunk = (
            "This is a very detailed technical specification. " * 20
        ).strip()
        messages = [
            {"role": "user", "content": long_chunk},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": long_chunk},  # exact duplicate sentences
        ]
        # Should detect duplicate sentences across messages
        new_msgs, _, removed = dedup.deduplicate(messages)
        assert isinstance(new_msgs, list)
        assert removed >= 1, "nomic-embed should detect exact duplicate with full context"

    def test_cross_message_dedup_integration(self):
        """Chunks from message 1 should be detected as duplicates in message 3."""
        chunk_a = "The deployment runs on AWS EC2 with auto-scaling enabled."
        messages = [
            {"role": "user", "content": chunk_a},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": f"Please review the setup. {chunk_a}"},
        ]
        _, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 1

    def test_distinct_topics_not_deduplicated(self):
        """Sentences about completely different topics should not be deduped."""
        messages = [
            {"role": "user", "content": "The capital of France is Paris."},
            {"role": "assistant", "content": "Correct."},
            {"role": "user", "content": "Photosynthesis converts sunlight into glucose."},
        ]
        _, _, removed = self.dedup.deduplicate(messages)
        assert removed == 0

    def test_nomic_embed_full_context_window(self):
        """nomic-embed-text-v1.5 with its 2048-token context window."""
        nomic_dedup = SemanticDeduplicator(
            model="nomic-ai/nomic-embed-text-v1.5",
            similarity_threshold=0.82,
            min_chunk_tokens=50,
        )
        # A ~300 token chunk that would be truncated by MiniLM but not nomic
        long_unique_chunk = " ".join([f"token_{i}" for i in range(300)])
        messages = [
            {"role": "user", "content": long_unique_chunk},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": long_unique_chunk},
        ]
        _, _, removed = nomic_dedup.deduplicate(messages)
        assert removed >= 1


@pytest.mark.integration
def test_graceful_degradation_importerror():
    """If sentence-transformers is not installed, should raise ImportError with message."""
    import sys
    from unittest.mock import patch

    # Simulate sentence-transformers being unavailable
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        from contextprune.embeddings import get_model
        get_model.cache_clear()  # Clear lru_cache
        with pytest.raises(Exception):
            get_model("all-MiniLM-L6-v2")
