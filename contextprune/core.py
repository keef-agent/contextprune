"""Core wrap() logic and middleware pipeline."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Union

from .budget import TokenBudgetInjector
from .dedup import SemanticDeduplicator
from .stats import CompressionStats, StatsTimer
from .tokenizer import count_message_tokens, count_system_tokens, count_tools_tokens
from .tool_filter import ToolSchemaFilter


@dataclasses.dataclass
class Config:
    """Configuration for the compression pipeline."""

    semantic_dedup: bool = True
    tool_filter: bool = True
    budget_injection: bool = True
    max_tools: int = 10
    similarity_threshold: float = 0.82
    min_chunk_tokens: int = 5
    dedup_model: str = "nomic-ai/nomic-embed-text-v1.5"
    tool_model: str = "nomic-ai/nomic-embed-text-v1.5"
    verbose: bool = False


def _extract_system_str(
    system: Optional[Union[str, List[Dict[str, Any]]]]
) -> Optional[str]:
    """Extract a plain string from a system prompt for dedup processing."""
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return " ".join(parts) if parts else None
    return None


class _CompressedMessages:
    """Proxy for the messages namespace that runs compression."""

    def __init__(self, original_messages: Any, config: Config) -> None:
        self._original = original_messages
        self._config = config
        self._dedup = SemanticDeduplicator(
            similarity_threshold=config.similarity_threshold,
            model=config.dedup_model,
            min_chunk_tokens=config.min_chunk_tokens,
        )
        self._tool_filter = ToolSchemaFilter(
            max_tools=config.max_tools,
            model=config.tool_model,
        )
        self._budget = TokenBudgetInjector()

    def create(self, **kwargs: Any) -> Any:
        """Intercept messages.create() and run compression."""
        timer = StatsTimer()
        stats = CompressionStats()

        with timer:
            messages = kwargs.get("messages", [])
            system = kwargs.get("system", None)
            tools = kwargs.get("tools", None)

            # Count original tokens
            stats.original_tokens = (
                count_message_tokens(messages)
                + count_system_tokens(system)
                + count_tools_tokens(tools)
            )

            # 1. Semantic deduplication
            if self._config.semantic_dedup and messages:
                system_str = _extract_system_str(system)
                new_messages, new_system_str, removed = self._dedup.deduplicate(
                    messages, system=system_str
                )
                kwargs["messages"] = new_messages
                stats.dedup_removed = removed

                # Update system if it was a string
                if system is not None and isinstance(system, str) and new_system_str:
                    kwargs["system"] = new_system_str

            # 2. Tool schema filtering
            if self._config.tool_filter and tools and len(tools) > self._config.max_tools:
                filtered_tools, removed = self._tool_filter.filter(
                    tools, kwargs.get("messages", messages)
                )
                kwargs["tools"] = filtered_tools
                stats.tools_removed = removed

            # 3. Token budget injection
            if self._config.budget_injection:
                current_system = kwargs.get("system", system)
                new_system, injected = self._budget.inject(
                    current_system, kwargs.get("messages", messages)
                )
                if injected:
                    kwargs["system"] = new_system
                    stats.budget_injected = True

            # Count compressed tokens
            stats.compressed_tokens = (
                count_message_tokens(kwargs.get("messages", messages))
                + count_system_tokens(kwargs.get("system", None))
                + count_tools_tokens(kwargs.get("tools", None))
            )

        stats.time_ms = round(timer.elapsed_ms, 2)
        if stats.original_tokens > 0:
            savings = (
                (stats.original_tokens - stats.compressed_tokens)
                / stats.original_tokens
                * 100
            )
            stats.savings_pct = round(max(0.0, savings), 1)

        if self._config.verbose:
            print(
                f"[contextprune] {stats.original_tokens} -> {stats.compressed_tokens} "
                f"tokens ({stats.savings_pct}% saved, {stats.time_ms}ms)"
            )

        # Call original
        response = self._original.create(**kwargs)

        # Attach stats to response
        response.compression_stats = stats  # type: ignore[attr-defined]
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class WrappedClient:
    """Wrapped Anthropic client with compression middleware."""

    def __init__(self, client: Any, config: Config) -> None:
        self._client = client
        self._config = config
        self.messages = _CompressedMessages(client.messages, config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap(
    client: Any,
    config: Optional[Config] = None,
    *,
    dedup_model: Optional[str] = None,
) -> WrappedClient:
    """Wrap an Anthropic client with compression middleware.

    Usage:
        import anthropic
        from contextprune import wrap

        client = wrap(anthropic.Anthropic())
        response = client.messages.create(...)
        print(response.compression_stats)

        # Use the lighter 22MB model (faster cold start):
        client = wrap(anthropic.Anthropic(), dedup_model="all-MiniLM-L6-v2")

    Args:
        client: An Anthropic client instance.
        config: Full Config object. Mutually usable with dedup_model shorthand.
        dedup_model: Shorthand to set the embedding model for both dedup and
            tool filter. Overrides config.dedup_model and config.tool_model.
    """
    if config is None:
        config = Config()
    if dedup_model is not None:
        config = dataclasses.replace(
            config, dedup_model=dedup_model, tool_model=dedup_model
        )
    return WrappedClient(client, config)


# --- OpenAI support ---


class _CompressedChatCompletions:
    """Proxy for OpenAI chat.completions that runs compression."""

    def __init__(self, original_completions: Any, config: Config) -> None:
        self._original = original_completions
        self._config = config
        self._dedup = SemanticDeduplicator(
            similarity_threshold=config.similarity_threshold,
            model=config.dedup_model,
            min_chunk_tokens=config.min_chunk_tokens,
        )
        self._tool_filter = ToolSchemaFilter(
            max_tools=config.max_tools,
            model=config.tool_model,
        )
        self._budget = TokenBudgetInjector()

    def create(self, **kwargs: Any) -> Any:
        """Intercept chat.completions.create() and run compression."""
        timer = StatsTimer()
        stats = CompressionStats()

        with timer:
            messages = kwargs.get("messages", [])
            tools = kwargs.get("tools", None)

            # Separate system messages from the rest
            system_msgs = [m for m in messages if m.get("role") == "system"]
            other_msgs = [m for m in messages if m.get("role") != "system"]

            system_str = None
            if system_msgs:
                system_str = " ".join(
                    m.get("content", "") for m in system_msgs
                    if isinstance(m.get("content", ""), str)
                )

            # Count original tokens
            stats.original_tokens = count_message_tokens(messages)
            if tools:
                stats.original_tokens += count_tools_tokens(tools)

            # 1. Semantic deduplication
            if self._config.semantic_dedup and other_msgs:
                new_msgs, new_system, removed = self._dedup.deduplicate(
                    other_msgs, system=system_str
                )
                stats.dedup_removed = removed

                # Rebuild messages list
                rebuilt = []
                if system_msgs and new_system:
                    rebuilt.append({"role": "system", "content": new_system})
                elif system_msgs:
                    rebuilt.extend(system_msgs)
                rebuilt.extend(new_msgs)
                kwargs["messages"] = rebuilt

            # 2. Tool filtering (OpenAI format)
            if self._config.tool_filter and tools and len(tools) > self._config.max_tools:
                # OpenAI tools are wrapped: {"type": "function", "function": {...}}
                unwrapped = []
                for t in tools:
                    if isinstance(t, dict) and "function" in t:
                        unwrapped.append(t["function"])
                    else:
                        unwrapped.append(t)

                filtered, removed = self._tool_filter.filter(
                    unwrapped, kwargs.get("messages", messages)
                )
                stats.tools_removed = removed

                # Re-wrap
                rewrapped = []
                for f in filtered:
                    rewrapped.append({"type": "function", "function": f})
                kwargs["tools"] = rewrapped

            # 3. Budget injection
            if self._config.budget_injection:
                current_msgs = kwargs.get("messages", messages)
                sys_msgs = [m for m in current_msgs if m.get("role") == "system"]
                if sys_msgs:
                    sys_content = sys_msgs[0].get("content", "")
                    non_sys = [m for m in current_msgs if m.get("role") != "system"]
                    new_sys, injected = self._budget.inject(sys_content, non_sys)
                    if injected:
                        rebuilt = [{"role": "system", "content": new_sys}]
                        rebuilt.extend(non_sys)
                        kwargs["messages"] = rebuilt
                        stats.budget_injected = True

            # Count compressed tokens
            stats.compressed_tokens = count_message_tokens(
                kwargs.get("messages", messages)
            )
            if kwargs.get("tools"):
                stats.compressed_tokens += count_tools_tokens(kwargs["tools"])

        stats.time_ms = round(timer.elapsed_ms, 2)
        if stats.original_tokens > 0:
            savings = (
                (stats.original_tokens - stats.compressed_tokens)
                / stats.original_tokens
                * 100
            )
            stats.savings_pct = round(max(0.0, savings), 1)

        if self._config.verbose:
            print(
                f"[contextprune] {stats.original_tokens} -> {stats.compressed_tokens} "
                f"tokens ({stats.savings_pct}% saved, {stats.time_ms}ms)"
            )

        response = self._original.create(**kwargs)
        response.compression_stats = stats  # type: ignore[attr-defined]
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _WrappedChat:
    """Proxy for OpenAI chat namespace."""

    def __init__(self, chat: Any, config: Config) -> None:
        self._chat = chat
        self._config = config
        self.completions = _CompressedChatCompletions(chat.completions, config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class WrappedOpenAIClient:
    """Wrapped OpenAI client with compression middleware."""

    def __init__(self, client: Any, config: Config) -> None:
        self._client = client
        self._config = config
        self.chat = _WrappedChat(client.chat, config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_openai(client: Any, config: Optional[Config] = None) -> WrappedOpenAIClient:
    """Wrap an OpenAI client with compression middleware.

    Usage:
        import openai
        from contextprune import wrap_openai

        client = wrap_openai(openai.OpenAI())
        response = client.chat.completions.create(...)
        print(response.compression_stats)
    """
    if config is None:
        config = Config()
    return WrappedOpenAIClient(client, config)
