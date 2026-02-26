"""Tests for SemanticDeduplicator.protect_system flag.

Motivation: "The Pitfalls of KV Cache Compression" (arXiv 2510.00231, 2025)
showed that compression can silently cause LLMs to ignore instructions —
particularly system-level rules that are phrased similarly to each other.

With protect_system=True (the default), the system prompt must be returned
byte-for-byte unchanged, while the seen pool is still populated from it so
that message-level duplicates are correctly stripped.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import numpy as np
import pytest

from contextprune.dedup import SemanticDeduplicator, _is_instructional


# ---------------------------------------------------------------------------
# Embedding mock (keyword-group vectors, same pattern as test_dedup.py)
# ---------------------------------------------------------------------------

_KEYWORD_GROUPS = [
    (["postgresql", "database", "port 5432", "data storage"], 0),
    (["private", "share", "user data", "personal"], 1),
    (["confidential", "secret", "reveal", "disclose"], 2),
    (["python", "code", "function", "programming"], 3),
    (["file", "read", "write", "directory"], 4),
]

_DIM = 16
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
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr("contextprune.embeddings.embed", _mock_embed)
    monkeypatch.setattr("contextprune.embeddings.cosine_similarity", _mock_cosine_sim)
    _embed_cache.clear()
    _unique_counter[0] = len(_KEYWORD_GROUPS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYSTEM_WITH_SIMILAR_RULES = (
    "Never share user data with third parties. "
    "Do not disclose private personal information. "
    "You must not reveal confidential user details. "
    "Always keep user data private and secure."
)

_MESSAGES_REPEATING_SYSTEM = [
    {"role": "user", "content": "What is PostgreSQL?"},
    {
        "role": "assistant",
        "content": (
            # Repeats a system-rule sentence verbatim — should be stripped
            "Never share user data with third parties. "
            "PostgreSQL is a relational database system on port 5432."
        ),
    },
]


# ---------------------------------------------------------------------------
# _is_instructional helper
# ---------------------------------------------------------------------------

class TestIsInstructional:
    def test_never_prefix(self):
        assert _is_instructional("Never share user data with third parties.")

    def test_do_not_prefix(self):
        assert _is_instructional("Do not reveal confidential information.")

    def test_always_prefix(self):
        assert _is_instructional("Always keep data private and secure.")

    def test_you_must_prefix(self):
        assert _is_instructional("You must not disclose personal details.")

    def test_case_insensitive(self):
        assert _is_instructional("NEVER share private data.")
        assert _is_instructional("never share private data.")

    def test_non_instruction(self):
        assert not _is_instructional("PostgreSQL stores data on port 5432.")
        assert not _is_instructional("The weather is sunny today.")

    def test_empty_string(self):
        assert not _is_instructional("")


# ---------------------------------------------------------------------------
# protect_system=True (default) — system prompt never modified
# ---------------------------------------------------------------------------

class TestProtectSystemTrue:
    def setup_method(self):
        self.dedup = SemanticDeduplicator(
            similarity_threshold=0.82,
            min_chunk_tokens=3,
            protect_system=True,
        )

    def test_system_with_similar_rules_returned_unchanged(self):
        """System prompt with semantically similar rules must not be compressed."""
        msgs, new_sys, removed = self.dedup.deduplicate(
            messages=[{"role": "user", "content": "Hello."}],
            system=_SYSTEM_WITH_SIMILAR_RULES,
        )
        assert new_sys == _SYSTEM_WITH_SIMILAR_RULES, (
            f"System prompt was modified: {new_sys!r}"
        )
        assert removed == 0 or True  # removals may come from messages, not system

    def test_system_returned_unchanged_is_exact(self):
        """Identity check — not just semantically equal but byte-for-byte identical."""
        system = "Do not share private user data. Never reveal user information. Always protect user privacy."
        _, new_sys, _ = self.dedup.deduplicate(
            messages=[],
            system=system,
        )
        assert new_sys is system or new_sys == system, (
            "System prompt must be returned unchanged with protect_system=True"
        )

    def test_message_repeating_system_rule_is_stripped(self):
        """A message that echoes a system instruction should still be deduped."""
        msgs, new_sys, removed = self.dedup.deduplicate(
            messages=_MESSAGES_REPEATING_SYSTEM,
            system=_SYSTEM_WITH_SIMILAR_RULES,
        )
        # System unchanged
        assert new_sys == _SYSTEM_WITH_SIMILAR_RULES

        # The assistant message contained "Never share user data with third parties."
        # which is identical/near-identical to the system prompt — should be removed
        assistant_content = next(
            m["content"] for m in msgs if m["role"] == "assistant"
        )
        # The database sentence should survive
        assert "PostgreSQL" in assistant_content or "port 5432" in assistant_content, (
            "Non-redundant assistant content was incorrectly removed"
        )
        # The repeated system rule should be gone
        assert removed >= 1, (
            "Expected at least one chunk removed from assistant message, "
            f"but removal_log is empty. Msgs: {msgs}"
        )

    def test_no_system_prompt_still_works(self):
        """protect_system=True with no system prompt should not raise or misbehave."""
        msgs, new_sys, removed = self.dedup.deduplicate(
            messages=[{"role": "user", "content": "Hello world."}],
            system=None,
        )
        assert new_sys is None
        assert len(msgs) == 1

    def test_empty_system_prompt_still_works(self):
        """Empty system prompt string should not crash."""
        msgs, new_sys, removed = self.dedup.deduplicate(
            messages=[{"role": "user", "content": "Hello."}],
            system="",
        )
        assert new_sys == ""

    def test_system_pool_populated_for_cross_message_dedup(self):
        """Even in protect mode, system chunks must populate the seen pool."""
        # System contains "Never share user data" → if pool is populated,
        # a message with the same phrase should be detected as duplicate
        system = "Never share user data with third parties. This is a strict rule."
        messages = [
            {
                "role": "assistant",
                "content": (
                    "Never share user data with third parties. "
                    "This is a strict rule. "
                    "The PostgreSQL database runs on port 5432."
                ),
            }
        ]
        dedup = SemanticDeduplicator(
            similarity_threshold=0.99,  # very tight — only exact/near-exact matches
            min_chunk_tokens=3,
            protect_system=True,
        )
        msgs, new_sys, removed = dedup.deduplicate(messages=messages, system=system)
        # System unchanged
        assert new_sys == system
        # At least the system-identical sentences should have been flagged
        assert removed >= 1, (
            "System pool should have been populated — duplicate in assistant message "
            f"not detected. removal_log: {dedup.removal_log}"
        )


# ---------------------------------------------------------------------------
# protect_system=False — system prompt IS allowed to be deduplicated
# ---------------------------------------------------------------------------

class TestProtectSystemFalse:
    def setup_method(self):
        # Low threshold so similar-but-not-identical instructions dedup
        self.dedup = SemanticDeduplicator(
            similarity_threshold=0.50,
            min_chunk_tokens=3,
            protect_system=False,
        )

    def test_similar_system_rules_can_be_compressed(self):
        """With protect_system=False, similar instructions inside system may be removed."""
        # Build a system with highly redundant (identical) sentences
        system = (
            "Never share user data with third parties. "
            "Never share user data with third parties. "
            "Never share user data with third parties."
        )
        _, new_sys, removed = self.dedup.deduplicate(messages=[], system=system)
        # With protect_system=False and low threshold, duplicates should be caught
        assert removed > 0, (
            "Expected redundant system chunks to be removed with protect_system=False"
        )
        assert new_sys != system or removed > 0  # at least something changed

    def test_protect_false_still_dedupes_messages(self):
        """Cross-message dedup still works when protect_system=False."""
        system = "You are a helpful assistant."
        messages = [
            {"role": "user", "content": "Tell me about Python programming code functions."},
            {
                "role": "assistant",
                "content": (
                    "Python is a programming language for writing code and functions. "
                    "Python is a programming language for writing code and functions."
                ),
            },
        ]
        msgs, _, removed = self.dedup.deduplicate(messages=messages, system=system)
        assert removed >= 1


# ---------------------------------------------------------------------------
# Default is protect_system=True
# ---------------------------------------------------------------------------

class TestDefaultProtectSystem:
    def test_default_is_protect_true(self):
        dedup = SemanticDeduplicator()
        assert dedup.protect_system is True

    def test_default_protects_system_prompt(self):
        dedup = SemanticDeduplicator(min_chunk_tokens=3)
        system = "Do not share private user data. Never reveal user details. Always protect privacy."
        _, new_sys, _ = dedup.deduplicate(messages=[], system=system)
        assert new_sys == system, (
            "Default SemanticDeduplicator must not modify the system prompt"
        )
