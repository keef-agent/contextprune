"""Unit tests for contextprune.adapters.openrouter.

All tests use mock OpenAI client — no real API calls.
"""

from __future__ import annotations

import sys
import types
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "/home/keith/contextprune")

from contextprune.adapters import SUPPORTED_MODELS, CompletionResult, OpenRouterAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(api_key: str = "sk-or-test") -> OpenRouterAdapter:
    """Return an OpenRouterAdapter with a mocked OpenAI client."""
    adapter = OpenRouterAdapter.__new__(OpenRouterAdapter)
    adapter.client = MagicMock()
    # Store api_key so _calculate_cost chain can reach it (used in exp5)
    adapter.client.api_key = api_key
    return adapter


def _mock_response(text: str = "4", input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    """Build a minimal mock chat completion response."""
    msg = MagicMock()
    msg.content = text

    choice = MagicMock()
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# 1. Alias resolution
# ---------------------------------------------------------------------------

class TestAliasResolution:
    def test_known_aliases_resolve(self):
        adapter = _make_adapter()
        assert adapter._resolve_model("claude") == "anthropic/claude-sonnet-4-6"
        assert adapter._resolve_model("gemini") == "google/gemini-3.1-pro-preview"
        assert adapter._resolve_model("grok") == "x-ai/grok-4.1-fast"
        assert adapter._resolve_model("kimi") == "moonshotai/kimi-k2.5"

    def test_full_model_id_passthrough(self):
        adapter = _make_adapter()
        full_id = "anthropic/claude-sonnet-4-6"
        assert adapter._resolve_model(full_id) == full_id

    def test_arbitrary_full_id_passthrough(self):
        adapter = _make_adapter()
        # Any ID with a slash should pass through without error
        assert adapter._resolve_model("openai/gpt-4o") == "openai/gpt-4o"

    def test_supported_models_dict_contents(self):
        assert "claude" in SUPPORTED_MODELS
        assert "gemini" in SUPPORTED_MODELS
        assert "grok" in SUPPORTED_MODELS
        assert "kimi" in SUPPORTED_MODELS
        assert "gpt52" in SUPPORTED_MODELS
        assert "codex" in SUPPORTED_MODELS
        assert len(SUPPORTED_MODELS) == 6


# ---------------------------------------------------------------------------
# 2. Unknown model raises ValueError with helpful message
# ---------------------------------------------------------------------------

class TestUnknownModelError:
    def test_unknown_alias_raises_value_error(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError, match="Unknown model"):
            adapter._resolve_model("unknownbot")

    def test_error_message_lists_valid_options(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError) as exc_info:
            adapter._resolve_model("notamodel")
        msg = str(exc_info.value)
        # Should list all 4 aliases
        for alias in SUPPORTED_MODELS:
            assert alias in msg

    def test_error_message_mentions_full_id_option(self):
        adapter = _make_adapter()
        with pytest.raises(ValueError) as exc_info:
            adapter._resolve_model("xyz")
        assert "full OpenRouter ID" in str(exc_info.value) or "/" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Cost calculation
# ---------------------------------------------------------------------------

class TestCostCalculation:
    def test_cost_claude(self):
        adapter = _make_adapter()
        # 1M input @ $3.00 + 1M output @ $15.00 = $18.00
        cost = adapter._calculate_cost("anthropic/claude-sonnet-4-6", 1_000_000, 1_000_000)
        assert abs(cost - 18.0) < 1e-6

    def test_cost_grok_cheap(self):
        adapter = _make_adapter()
        # 100k input @ $0.20/M + 100k output @ $0.50/M
        # = 0.10 * 0.20 + 0.10 * 0.50 = 0.02 + 0.05 = 0.07
        cost = adapter._calculate_cost("x-ai/grok-4.1-fast", 100_000, 100_000)
        assert abs(cost - 0.07) < 1e-6

    def test_cost_zero_tokens(self):
        adapter = _make_adapter()
        cost = adapter._calculate_cost("anthropic/claude-sonnet-4-6", 0, 0)
        assert cost == 0.0

    def test_cost_unknown_model_warns_and_returns_zero(self):
        adapter = _make_adapter()
        with pytest.warns(UserWarning, match="No pricing data"):
            cost = adapter._calculate_cost("unknown/model", 1000, 500)
        assert cost == 0.0

    def test_cost_kimi(self):
        adapter = _make_adapter()
        # 500k input @ $0.45/M + 200k output @ $2.20/M
        # = 0.5 * 0.45 + 0.2 * 2.20 = 0.225 + 0.44 = 0.665
        cost = adapter._calculate_cost("moonshotai/kimi-k2.5", 500_000, 200_000)
        assert abs(cost - 0.665) < 1e-6


# ---------------------------------------------------------------------------
# 4. CompletionResult dataclass structure
# ---------------------------------------------------------------------------

class TestCompletionResultDataclass:
    def test_all_fields_present(self):
        field_names = {f.name for f in fields(CompletionResult)}
        assert field_names == {
            "text", "input_tokens", "output_tokens",
            "model", "latency_ms", "cost_usd"
        }

    def test_construct_directly(self):
        r = CompletionResult(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            model="anthropic/claude-sonnet-4-6",
            latency_ms=123.4,
            cost_usd=0.00005,
        )
        assert r.text == "hello"
        assert r.input_tokens == 10
        assert r.output_tokens == 5
        assert r.model == "anthropic/claude-sonnet-4-6"
        assert r.latency_ms == 123.4
        assert r.cost_usd == 0.00005

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(CompletionResult)


# ---------------------------------------------------------------------------
# 5. complete() end-to-end with mocked client
# ---------------------------------------------------------------------------

class TestCompleteMethod:
    def test_basic_completion(self):
        adapter = _make_adapter()
        adapter.client.chat.completions.create.return_value = _mock_response(
            text="4", input_tokens=12, output_tokens=3
        )
        result = adapter.complete(
            messages=[{"role": "user", "content": "2+2?"}],
            model="grok",
        )
        assert isinstance(result, CompletionResult)
        assert result.text == "4"
        assert result.input_tokens == 12
        assert result.output_tokens == 3
        assert result.model == "x-ai/grok-4.1-fast"
        assert result.latency_ms >= 0
        assert result.cost_usd >= 0

    def test_system_prepended(self):
        adapter = _make_adapter()
        adapter.client.chat.completions.create.return_value = _mock_response()

        adapter.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude",
            system="Be terse.",
        )

        call_kwargs = adapter.client.chat.completions.create.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "Be terse."}
        assert msgs[1] == {"role": "user", "content": "hi"}

    def test_no_system_no_prepend(self):
        adapter = _make_adapter()
        adapter.client.chat.completions.create.return_value = _mock_response()

        adapter.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="kimi",
        )

        call_kwargs = adapter.client.chat.completions.create.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "user"

    def test_temperature_default_zero(self):
        adapter = _make_adapter()
        adapter.client.chat.completions.create.return_value = _mock_response()
        adapter.complete(messages=[{"role": "user", "content": "x"}], model="gemini")
        call_kwargs = adapter.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_model_id_passed_to_api(self):
        adapter = _make_adapter()
        adapter.client.chat.completions.create.return_value = _mock_response()
        adapter.complete(messages=[{"role": "user", "content": "x"}], model="kimi")
        call_kwargs = adapter.client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "moonshotai/kimi-k2.5"

    def test_cost_computed_from_usage(self):
        adapter = _make_adapter()
        # 1M input tokens @ $0.20/M for grok = $0.20, 0 output = $0.00
        adapter.client.chat.completions.create.return_value = _mock_response(
            input_tokens=1_000_000, output_tokens=0
        )
        result = adapter.complete(
            messages=[{"role": "user", "content": "x"}],
            model="grok",
        )
        assert abs(result.cost_usd - 0.20) < 1e-6
