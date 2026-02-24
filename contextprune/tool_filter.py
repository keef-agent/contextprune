"""Tool schema filtering using embedding-based relevance scoring.

Given N tools in the schema, selects the K most semantically relevant tools
for the current user message using sentence-transformer embeddings.

Default model: nomic-ai/nomic-embed-text-v1.5 (shared with SemanticDeduplicator).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_FALLBACK_MODEL = "all-MiniLM-L6-v2"

# nomic-embed-text-v1.5 task prefixes
_QUERY_PREFIX = "search_query: "
_DOC_PREFIX = "search_document: "


def _require_embeddings():
    """Import embeddings module with a helpful error on missing deps."""
    try:
        from . import embeddings
        return embeddings
    except ImportError:
        pass
    try:
        import contextprune.embeddings as embeddings
        return embeddings
    except ImportError:
        pass
    raise ImportError(
        "sentence-transformers is required for ToolSchemaFilter. "
        "Install it with: pip install sentence-transformers"
    )


class ToolSchemaFilter:
    """Filter tool schemas to include only the most relevant tools.

    Uses embedding-based semantic similarity to score each tool against the
    user's most recent message. Returns the top K highest-scoring tools.

    Algorithm:
        1. Extract the user's most recent message as the query.
        2. Build a text representation for each tool (name + description).
        3. Embed query (with "search_query: " prefix) and all tool representations
           (with "search_document: " prefix) in a single batch per group.
        4. Score each tool by cosine similarity to the query embedding.
        5. Return top max_tools tools sorted by original schema order.
        6. If no user messages or query is empty: return all tools (safe fallback).

    Attributes:
        score_log: List of (tool_name, score, "kept"/"dropped") tuples,
            populated after each call to filter(). Useful for debugging
            and recall evaluation.
    """

    def __init__(
        self,
        max_tools: int = 10,
        model: str = _DEFAULT_MODEL,
        score_by: str = "description",
        fallback_to_all: bool = True,
    ) -> None:
        """Initialize the tool filter.

        Args:
            max_tools: Maximum number of tools to keep. Default: 10.
            model: HuggingFace model name. Default: nomic-ai/nomic-embed-text-v1.5.
                Use "minilm" or "all-MiniLM-L6-v2" for the fast 22MB fallback.
            score_by: How to build tool text for scoring:
                - "description": Use tool name + description (default).
                - "name_and_description": Expand snake_case name + description.
            fallback_to_all: If embedding fails (e.g., model unavailable), return
                all tools rather than raising. Default: True.
        """
        if model == "minilm":
            model = _FALLBACK_MODEL
        self.max_tools = max_tools
        self.model = model
        self.score_by = score_by
        self.fallback_to_all = fallback_to_all
        self.score_log: List[Tuple[str, float, str]] = []

    def _tool_text(self, tool: Dict[str, Any]) -> str:
        """Build a searchable text representation for a tool."""
        name = tool.get("name", "")
        desc = tool.get("description", "")
        if self.score_by == "name_and_description":
            # Expand snake_case and camelCase for better semantic matching
            expanded = re.sub(r"[_-]", " ", name)
            expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", expanded)
            return f"{expanded} {name} {desc}".strip()
        else:
            # Default: name + description (name helps with direct noun matching)
            return f"{name} {desc}".strip()

    def _extract_query(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the most recent user message content as the query string."""
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return None

        last_user = user_messages[-1]
        content = last_user.get("content", "")

        if isinstance(content, str):
            return content.strip() or None

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
            result = " ".join(parts).strip()
            return result or None

        return None

    def filter(
        self,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Filter tools to the most semantically relevant ones.

        Args:
            tools: List of tool schema dicts with "name" and "description" keys.
            messages: Conversation messages used to extract the user query.

        Returns:
            Tuple of (filtered_tools, num_removed):
                - filtered_tools: Top max_tools tools in their original schema order.
                - num_removed: Number of tools removed.
        """
        self.score_log = []

        if len(tools) <= self.max_tools:
            return tools, 0

        # Extract query from most recent user message
        query = self._extract_query(messages)
        if not query:
            # No user message or empty query — safe fallback: return all
            return tools, 0

        try:
            emb = _require_embeddings()
        except ImportError:
            if self.fallback_to_all:
                return tools, 0
            raise

        try:
            import numpy as np

            # Embed the user query
            query_embs = emb.embed([query], self.model, prefix=_QUERY_PREFIX)
            query_emb = query_embs[0]

            # Batch embed all tool representations
            tool_texts = [self._tool_text(t) for t in tools]
            tool_embs = emb.embed(tool_texts, self.model, prefix=_DOC_PREFIX)

            # Score each tool by cosine similarity to the query
            scored: List[Tuple[float, int, Dict[str, Any]]] = []
            for i, (tool, tool_emb) in enumerate(zip(tools, tool_embs)):
                score = emb.cosine_similarity(query_emb, tool_emb)
                scored.append((score, i, tool))

            # Sort by score descending, original index ascending for ties
            scored.sort(key=lambda x: (-x[0], x[1]))

            # Build score log
            kept_indices = {item[1] for item in scored[: self.max_tools]}
            for score, idx, tool in scored:
                status = "kept" if idx in kept_indices else "dropped"
                self.score_log.append((tool.get("name", ""), score, status))

            # Take top K and restore original schema order
            selected = sorted(scored[: self.max_tools], key=lambda x: x[1])
            filtered = [item[2] for item in selected]
            removed = len(tools) - len(filtered)
            return filtered, removed

        except Exception:
            if self.fallback_to_all:
                return tools, 0
            raise
