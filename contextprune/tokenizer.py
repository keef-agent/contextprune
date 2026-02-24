"""Lightweight token counting.

Uses tiktoken if available, otherwise falls back to a character-based estimate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import json

_encoder: Optional[Any] = None
_USE_TIKTOKEN = False

try:
    import tiktoken

    _encoder = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except ImportError:
    pass


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    if _USE_TIKTOKEN and _encoder is not None:
        return len(_encoder.encode(text))
    # Rough estimate: 1 token per 4 characters
    return max(1, len(text) // 4)


def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Count tokens across a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total += count_tokens(text)
        # Role overhead
        total += 4
    return total


def count_tools_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    """Count tokens in tool schemas."""
    if not tools:
        return 0
    return count_tokens(json.dumps(tools))


def count_system_tokens(system: Optional[Union[str, List[Dict[str, Any]]]]) -> int:
    """Count tokens in a system prompt."""
    if system is None:
        return 0
    if isinstance(system, str):
        return count_tokens(system)
    if isinstance(system, list):
        total = 0
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    total += count_tokens(text)
        return total
    return 0
