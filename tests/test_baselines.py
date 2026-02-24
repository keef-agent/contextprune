"""Tests for contextprune.baselines.LLMLingua2Baseline.

Test categories:
  - Unit tests (no model download): interface contract, ImportError, stats structure
  - Integration tests (@pytest.mark.integration): real model, real compression

Run unit tests only (fast):
    pytest tests/test_baselines.py -v -m "not integration"

Run all including integration (requires llmlingua + model download ~500MB):
    pytest tests/test_baselines.py -v
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/home/keith/contextprune")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES: List[Dict[str, Any]] = [
    {"role": "user", "content": "Explain quantum computing in simple terms."},
    {"role": "assistant", "content": "Quantum computing uses quantum bits (qubits) that can be in a superposition of 0 and 1 simultaneously, unlike classical bits. This allows quantum computers to explore many solutions in parallel. Key concepts include superposition, entanglement (qubits can be correlated regardless of distance), and interference (amplifying correct paths and cancelling incorrect ones). For certain problems like factoring large numbers or searching unsorted databases, quantum computers offer exponential or quadratic speedups over classical computers."},
    {"role": "user", "content": "What are the main practical applications today?"},
]

SAMPLE_SYSTEM = (
    "You are an expert physics and computer science tutor. "
    "Explain complex concepts clearly and accurately. "
    "Always provide concrete examples and analogies. "
    "Be concise but thorough. Use simple language for non-experts."
)


def make_mock_compressor(
    compressed_prompt: str = "compressed text",
    origin_tokens: int = 100,
    compressed_tokens: int = 50,
) -> MagicMock:
    """Create a mock PromptCompressor that returns controllable results."""
    mock = MagicMock()
    mock.compress_prompt.return_value = {
        "compressed_prompt": compressed_prompt,
        "origin_tokens": origin_tokens,
        "compressed_tokens": compressed_tokens,
    }
    return mock


# ---------------------------------------------------------------------------
# Unit tests — no model download required
# ---------------------------------------------------------------------------

class TestLLMLingua2BaselineImportError:
    """Test that missing llmlingua raises a helpful ImportError."""

    def test_raises_import_error_when_llmlingua_not_installed(self):
        """ImportError with helpful install message when llmlingua is missing."""
        from contextprune.baselines.llmlingua2 import _get_compressor

        # Temporarily block the llmlingua import by patching builtins.__import__
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "llmlingua":
                raise ImportError("No module named 'llmlingua'")
            return real_import(name, *args, **kwargs)

        # Clear the lru_cache so the mocked import is actually called
        _get_compressor.cache_clear()

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pip install llmlingua"):
                from contextprune.baselines.llmlingua2 import _get_compressor as gc
                gc.cache_clear()
                gc("some-model", "cpu")

        # Restore cache state for other tests
        _get_compressor.cache_clear()

    def test_import_error_message_is_helpful(self):
        """The ImportError message should tell the user exactly what to do."""
        from contextprune.baselines.llmlingua2 import _get_compressor

        _get_compressor.cache_clear()

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "llmlingua":
                raise ImportError("No module named 'llmlingua'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            try:
                from contextprune.baselines.llmlingua2 import _get_compressor as gc
                gc.cache_clear()
                gc("test-model", "cpu")
                assert False, "Should have raised ImportError"
            except ImportError as e:
                error_msg = str(e)
                assert "llmlingua" in error_msg.lower(), f"Should mention llmlingua: {error_msg}"
                assert "pip install" in error_msg.lower(), f"Should mention pip install: {error_msg}"

        _get_compressor.cache_clear()


class TestLLMLingua2BaselineInit:
    """Test __init__ parameter validation."""

    def test_default_params(self):
        from contextprune.baselines import LLMLingua2Baseline
        b = LLMLingua2Baseline()
        assert b.rate == 0.5
        assert b.device == "cpu"
        assert b.force_tokens == ["\n"]

    def test_custom_rate(self):
        from contextprune.baselines import LLMLingua2Baseline
        b = LLMLingua2Baseline(rate=0.3)
        assert b.rate == 0.3

    def test_invalid_rate_zero(self):
        from contextprune.baselines import LLMLingua2Baseline
        with pytest.raises(ValueError, match="rate"):
            LLMLingua2Baseline(rate=0.0)

    def test_invalid_rate_negative(self):
        from contextprune.baselines import LLMLingua2Baseline
        with pytest.raises(ValueError, match="rate"):
            LLMLingua2Baseline(rate=-0.1)

    def test_invalid_rate_above_one(self):
        from contextprune.baselines import LLMLingua2Baseline
        with pytest.raises(ValueError, match="rate"):
            LLMLingua2Baseline(rate=1.5)

    def test_rate_one_is_valid(self):
        """rate=1.0 means keep 100% (no compression), should not raise."""
        from contextprune.baselines import LLMLingua2Baseline
        b = LLMLingua2Baseline(rate=1.0)
        assert b.rate == 1.0

    def test_custom_force_tokens(self):
        from contextprune.baselines import LLMLingua2Baseline
        b = LLMLingua2Baseline(force_tokens=["\n", "."])
        assert b.force_tokens == ["\n", "."]

    def test_none_force_tokens_defaults_to_newline(self):
        from contextprune.baselines import LLMLingua2Baseline
        b = LLMLingua2Baseline(force_tokens=None)
        assert b.force_tokens == ["\n"]


class TestLLMLingua2BaselineCompressInterface:
    """Test compress() return type and stats structure — using a mocked compressor."""

    def _make_baseline_with_mock(self, rate: float = 0.5):
        """Return a LLMLingua2Baseline with its compressor mocked out."""
        from contextprune.baselines import LLMLingua2Baseline
        baseline = LLMLingua2Baseline(rate=rate)
        mock_compressor = make_mock_compressor(
            compressed_prompt="short text",
            origin_tokens=200,
            compressed_tokens=100,
        )
        baseline._load_compressor = lambda: mock_compressor
        return baseline, mock_compressor

    def test_returns_three_tuple(self):
        baseline, _ = self._make_baseline_with_mock()
        result = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_compressed_messages_is_list(self):
        baseline, _ = self._make_baseline_with_mock()
        msgs, sys_out, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert isinstance(msgs, list)

    def test_compressed_system_is_str_or_none(self):
        baseline, _ = self._make_baseline_with_mock()
        _, sys_out, _ = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert sys_out is None or isinstance(sys_out, str)

    def test_compressed_system_none_when_input_none(self):
        baseline, _ = self._make_baseline_with_mock()
        _, sys_out, _ = baseline.compress(SAMPLE_MESSAGES, system=None)
        assert sys_out is None

    def test_stats_is_dict(self):
        baseline, _ = self._make_baseline_with_mock()
        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert isinstance(stats, dict)

    def test_stats_has_required_keys(self):
        baseline, _ = self._make_baseline_with_mock()
        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        required_keys = {
            "original_tokens",
            "compressed_tokens",
            "reduction_pct",
            "method",
            "rate",
        }
        missing = required_keys - set(stats.keys())
        assert not missing, f"Stats missing keys: {missing}"

    def test_stats_method_is_llmlingua2(self):
        baseline, _ = self._make_baseline_with_mock()
        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert stats["method"] == "llmlingua2"

    def test_stats_rate_matches_init_rate(self):
        baseline, _ = self._make_baseline_with_mock(rate=0.3)
        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert stats["rate"] == 0.3

    def test_stats_tokens_are_integers(self):
        baseline, _ = self._make_baseline_with_mock()
        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert isinstance(stats["original_tokens"], int)
        assert isinstance(stats["compressed_tokens"], int)

    def test_stats_reduction_pct_is_float(self):
        baseline, _ = self._make_baseline_with_mock()
        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert isinstance(stats["reduction_pct"], float)

    def test_stats_reduction_pct_non_negative(self):
        baseline, _ = self._make_baseline_with_mock()
        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert stats["reduction_pct"] >= 0.0

    def test_message_roles_preserved(self):
        """Role labels must survive compression."""
        baseline, _ = self._make_baseline_with_mock()
        original_roles = [m["role"] for m in SAMPLE_MESSAGES]
        compressed_msgs, _, _ = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        compressed_roles = [m["role"] for m in compressed_msgs]
        assert compressed_roles == original_roles

    def test_message_count_preserved(self):
        """Number of messages must not change after compression."""
        baseline, _ = self._make_baseline_with_mock()
        compressed_msgs, _, _ = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)
        assert len(compressed_msgs) == len(SAMPLE_MESSAGES)

    def test_empty_messages_handled(self):
        """compress() must not crash on empty message list."""
        baseline, _ = self._make_baseline_with_mock()
        msgs, sys_out, stats = baseline.compress([], SAMPLE_SYSTEM)
        assert isinstance(msgs, list)
        assert len(msgs) == 0

    def test_empty_system_handled(self):
        """compress() must not crash when system is empty string."""
        baseline, _ = self._make_baseline_with_mock()
        msgs, sys_out, stats = baseline.compress(SAMPLE_MESSAGES, "")
        assert isinstance(msgs, list)

    def test_non_string_content_passes_through(self):
        """Messages with non-string content (e.g. tool results) are not modified."""
        from contextprune.baselines import LLMLingua2Baseline
        baseline = LLMLingua2Baseline()
        mock_compressor = make_mock_compressor(
            compressed_prompt="compressed", origin_tokens=50, compressed_tokens=25
        )
        baseline._load_compressor = lambda: mock_compressor

        messages_with_tool = [
            {"role": "user", "content": "Run this tool"},
            {"role": "tool", "content": [{"type": "tool_result", "content": "result"}]},
        ]
        msgs, _, _ = baseline.compress(messages_with_tool, None)
        # Tool message should pass through unchanged
        assert msgs[1]["content"] == messages_with_tool[1]["content"]


class TestLLMLingua2BaselineRedistribute:
    """Test the internal _redistribute_compressed logic."""

    def test_redistribute_returns_same_count(self):
        from contextprune.baselines.llmlingua2 import LLMLingua2Baseline
        b = LLMLingua2Baseline()
        result = b._redistribute_compressed(SAMPLE_MESSAGES, "some compressed output text here")
        assert len(result) == len(SAMPLE_MESSAGES)

    def test_redistribute_preserves_roles(self):
        from contextprune.baselines.llmlingua2 import LLMLingua2Baseline
        b = LLMLingua2Baseline()
        result = b._redistribute_compressed(SAMPLE_MESSAGES, "compressed text content")
        for orig, new in zip(SAMPLE_MESSAGES, result):
            assert new["role"] == orig["role"]

    def test_redistribute_empty_messages(self):
        from contextprune.baselines.llmlingua2 import LLMLingua2Baseline
        b = LLMLingua2Baseline()
        result = b._redistribute_compressed([], "text")
        assert result == []


# ---------------------------------------------------------------------------
# Integration tests — requires llmlingua + real model download (~500MB)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLLMLingua2BaselineIntegration:
    """Real model tests. Require llmlingua to be installed and model downloaded."""

    @pytest.fixture(autouse=True)
    def check_llmlingua(self):
        """Skip all integration tests if llmlingua is not installed."""
        try:
            import llmlingua  # noqa: F401
        except ImportError:
            pytest.skip("llmlingua not installed — run: pip install llmlingua")

    def test_compress_reduces_tokens(self):
        """Compressed output should have fewer tokens than the original."""
        from contextprune.baselines import LLMLingua2Baseline
        from contextprune.tokenizer import count_message_tokens, count_system_tokens

        baseline = LLMLingua2Baseline(rate=0.5)
        orig_tokens = count_message_tokens(SAMPLE_MESSAGES) + count_system_tokens(SAMPLE_SYSTEM)

        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)

        assert stats["compressed_tokens"] < stats["original_tokens"], (
            f"Expected compression: {stats['original_tokens']} → {stats['compressed_tokens']}"
        )
        assert stats["original_tokens"] > 0

    def test_compress_stats_are_consistent(self):
        """reduction_pct should be consistent with token counts."""
        from contextprune.baselines import LLMLingua2Baseline

        baseline = LLMLingua2Baseline(rate=0.5)
        _, _, stats = baseline.compress(SAMPLE_MESSAGES, SAMPLE_SYSTEM)

        orig = stats["original_tokens"]
        comp = stats["compressed_tokens"]
        expected_pct = (orig - comp) / orig * 100 if orig else 0
        assert abs(stats["reduction_pct"] - expected_pct) < 1.0, (
            f"reduction_pct inconsistency: computed {expected_pct:.1f}% vs stated {stats['reduction_pct']:.1f}%"
        )

    def test_compress_roles_intact(self):
        """Role labels must survive real model compression."""
        from contextprune.baselines import LLMLingua2Baseline

        baseline = LLMLingua2Baseline(rate=0.5)
        original_roles = [m["role"] for m in SAMPLE_MESSAGES]
        compressed_msgs, _, _ = baseline.compress(SAMPLE_MESSAGES)
        assert [m["role"] for m in compressed_msgs] == original_roles

    def test_model_caching(self):
        """Second call to _load_compressor() should be near-instant (<100ms)."""
        import time
        from contextprune.baselines import LLMLingua2Baseline

        baseline = LLMLingua2Baseline(rate=0.5)
        baseline._load_compressor()  # warm the cache

        t0 = time.perf_counter()
        baseline._load_compressor()
        t1 = time.perf_counter()

        assert (t1 - t0) * 1000 < 100, "Cached model load should be <100ms"

    def test_different_rates_produce_different_lengths(self):
        """rate=0.3 should produce shorter output than rate=0.7."""
        from contextprune.baselines import LLMLingua2Baseline
        from contextprune.baselines.llmlingua2 import _get_compressor

        # Use a long enough input that compression differences show up clearly
        long_content = (
            "The quantum computer uses qubits that can be in superposition states. "
            "Unlike classical bits which are either 0 or 1, qubits can be both simultaneously. "
            "Entanglement allows qubits to be correlated regardless of physical distance. "
            "Interference amplifies correct computational paths while cancelling incorrect ones. "
            "Quantum algorithms like Shor's algorithm can factor large integers exponentially faster. "
            "Grover's algorithm provides quadratic speedup for searching unsorted databases. "
            "Current quantum computers are called NISQ (Noisy Intermediate-Scale Quantum) devices. "
            "Error correction is a major challenge due to qubit decoherence and gate errors. "
        ) * 3

        msgs = [{"role": "user", "content": long_content}]

        b30 = LLMLingua2Baseline(rate=0.3)
        b70 = LLMLingua2Baseline(rate=0.7)

        _, _, stats30 = b30.compress(msgs)
        _, _, stats70 = b70.compress(msgs)

        assert stats30["compressed_tokens"] <= stats70["compressed_tokens"], (
            f"rate=0.3 should compress more than rate=0.7: "
            f"{stats30['compressed_tokens']} vs {stats70['compressed_tokens']}"
        )
