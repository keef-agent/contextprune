"""
Multi-model adapter via OpenRouter.
Provides a unified interface for running experiments across model families.

Supported models (as of 2026-02):
  - anthropic/claude-sonnet-4-6
  - google/gemini-3.1-pro-preview
  - x-ai/grok-4.1-fast
  - moonshotai/kimi-k2.5
  - openai/gpt-5.2
  - openai/gpt-5.3-codex
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

SUPPORTED_MODELS: dict[str, str] = {
    "claude":  "anthropic/claude-sonnet-4-6",
    "gemini":  "google/gemini-3.1-pro-preview",
    "grok":    "x-ai/grok-4.1-fast",
    "kimi":    "moonshotai/kimi-k2.5",
    "gpt52":   "openai/gpt-5.2",
    "codex":   "openai/gpt-5.3-codex",
}


@dataclass
class CompletionResult:
    text: str          # model output text
    input_tokens: int  # tokens consumed (input)
    output_tokens: int # tokens generated (output)
    model: str         # model ID used
    latency_ms: float  # wall-clock time for the API call
    cost_usd: float    # estimated cost based on OpenRouter pricing


class OpenRouterAdapter:
    """
    Unified completion interface across model families via OpenRouter.

    Usage:
        adapter = OpenRouterAdapter(api_key="sk-or-...")
        result = adapter.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude",   # or full ID: "anthropic/claude-sonnet-4-6"
            system="You are helpful.",
            max_tokens=512,
        )
        print(result.text, result.cost_usd)
    """

    # Pricing per million tokens (as of 2026-02)
    PRICING: dict[str, dict[str, float]] = {
        "anthropic/claude-sonnet-4-6":   {"input":  3.00, "output": 15.00},
        "google/gemini-3.1-pro-preview":  {"input":  2.00, "output": 12.00},
        "x-ai/grok-4.1-fast":            {"input":  0.20, "output":  0.50},
        "moonshotai/kimi-k2.5":          {"input":  0.45, "output":  2.20},
        "openai/gpt-5.2":                {"input":  1.75, "output": 14.00},
        "openai/gpt-5.3-codex":          {"input":  1.75, "output": 14.00},
    }

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def _resolve_model(self, model: str) -> str:
        """Resolve a short alias to a full OpenRouter model ID."""
        if model in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model]
        # Accept any full model ID that contains a slash (provider/model)
        if "/" in model:
            return model
        valid = ", ".join(f'"{k}" → {v}' for k, v in SUPPORTED_MODELS.items())
        raise ValueError(
            f"Unknown model {model!r}. Valid aliases: {valid}. "
            f"You may also pass a full OpenRouter ID (e.g. 'anthropic/claude-sonnet-4-6')."
        )

    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD from token counts and per-million pricing."""
        pricing = self.PRICING.get(model_id)
        if pricing is None:
            warnings.warn(
                f"No pricing data for model {model_id!r}. Returning cost=0.0.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0
        cost = (input_tokens / 1_000_000) * pricing["input"]
        cost += (output_tokens / 1_000_000) * pricing["output"]
        return round(cost, 8)

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str = "claude",
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,  # deterministic by default for benchmarks
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Run a completion. Returns CompletionResult with text, tokens, latency, cost.

        model: short alias ("claude", "gemini", "grok", "kimi") or full OpenRouter ID.
        system: system prompt string (prepended as system message if provided).
        temperature: 0.0 by default for reproducible benchmark results.
        """
        model_id = self._resolve_model(model)

        # Build message list
        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Retry on 429 with exponential backoff
        max_retries = 3
        retry_delays = [2, 4, 8]
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                latency_ms = (time.perf_counter() - t0) * 1000

                text = response.choices[0].message.content or ""
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                cost = self._calculate_cost(model_id, input_tokens, output_tokens)

                return CompletionResult(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model_id,
                    latency_ms=round(latency_ms, 2),
                    cost_usd=cost,
                )

            except Exception as exc:
                last_exc = exc
                # Check for rate limit (openai SDK wraps as RateLimitError or HTTPStatusError)
                is_rate_limit = (
                    "429" in str(exc)
                    or "rate_limit" in str(exc).lower()
                    or type(exc).__name__ in ("RateLimitError",)
                )
                if is_rate_limit and attempt < max_retries:
                    delay = retry_delays[attempt]
                    warnings.warn(
                        f"Rate limited on {model_id!r} (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay}s…",
                        UserWarning,
                        stacklevel=2,
                    )
                    time.sleep(delay)
                    continue
                raise

        # Should not reach here, but satisfy type checker
        raise RuntimeError(f"All retries exhausted for {model_id!r}") from last_exc
