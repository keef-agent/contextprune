"""Tool schema filtering.

Given N tools in the schema, selects the K most relevant to the current user
message based on keyword matching between message content and tool
names/descriptions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _extract_keywords(text: str) -> List[str]:
    """Extract lowercase keywords from text."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def _tool_text(tool: Dict[str, Any]) -> str:
    """Extract searchable text from a tool schema."""
    parts = []
    name = tool.get("name", "")
    if name:
        # Split snake_case and camelCase into words
        expanded = re.sub(r"[_-]", " ", name)
        expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", expanded)
        parts.append(expanded)
        parts.append(name)

    desc = tool.get("description", "")
    if desc:
        parts.append(desc)

    # Include parameter names
    input_schema = tool.get("input_schema", {})
    if isinstance(input_schema, dict):
        props = input_schema.get("properties", {})
        if isinstance(props, dict):
            for prop_name in props:
                expanded = re.sub(r"[_-]", " ", prop_name)
                parts.append(expanded)

    return " ".join(parts)


def _score_tool(tool: Dict[str, Any], message_keywords: Counter) -> float:
    """Score a tool's relevance to the message keywords."""
    tool_text = _tool_text(tool)
    tool_keywords = _extract_keywords(tool_text)

    if not tool_keywords or not message_keywords:
        return 0.0

    score = 0.0
    tool_keyword_set = set(tool_keywords)
    for keyword, count in message_keywords.items():
        if keyword in tool_keyword_set:
            score += count
        # Partial match: keyword is substring of a tool keyword or vice versa
        else:
            for tk in tool_keyword_set:
                if len(keyword) >= 3 and (keyword in tk or tk in keyword):
                    score += count * 0.5
                    break

    return score


class ToolSchemaFilter:
    """Filter tool schemas to only include relevant tools."""

    def __init__(self, max_tools: int = 10) -> None:
        self.max_tools = max_tools

    def filter(
        self,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Filter tools to the most relevant ones.

        Returns (filtered_tools, num_removed).
        """
        if len(tools) <= self.max_tools:
            return tools, 0

        # Get keywords from recent messages (last 3 user messages)
        message_keywords: Counter = Counter()
        user_messages = [m for m in messages if m.get("role") == "user"]
        recent = user_messages[-3:] if len(user_messages) > 3 else user_messages

        for msg in recent:
            content = msg.get("content", "")
            if isinstance(content, str):
                message_keywords.update(_extract_keywords(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if text:
                            message_keywords.update(_extract_keywords(text))

        # Score each tool
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for i, tool in enumerate(tools):
            score = _score_tool(tool, message_keywords)
            scored.append((score, i, tool))

        # Sort by score descending, then by original order for ties
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Take top K
        selected = scored[: self.max_tools]
        # Restore original order
        selected.sort(key=lambda x: x[1])

        filtered = [item[2] for item in selected]
        removed = len(tools) - len(filtered)
        return filtered, removed
