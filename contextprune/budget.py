"""Token budget injection.

Appends a calibrated token budget hint to the system prompt based on task
complexity estimation. Implements the TALE paper pattern.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .tokenizer import count_tokens


def _estimate_complexity(messages: List[Dict[str, Any]]) -> str:
    """Estimate task complexity from messages.

    Returns "low", "medium", or "high".
    """
    if not messages:
        return "low"

    # Get the last user message
    last_user = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user = content
            elif isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        texts.append(block["text"])
                last_user = " ".join(texts)
            break

    if not last_user:
        return "low"

    tokens = count_tokens(last_user)

    # Complexity signals
    complexity_keywords = [
        "explain", "analyze", "compare", "implement", "design",
        "architecture", "refactor", "debug", "comprehensive", "detailed",
        "step by step", "walkthrough", "tutorial",
    ]
    keyword_count = sum(
        1 for kw in complexity_keywords if kw in last_user.lower()
    )

    has_code_request = bool(
        re.search(r"(write|create|build|implement|code)", last_user.lower())
    )

    if tokens > 200 or keyword_count >= 3:
        return "high"
    elif tokens > 50 or keyword_count >= 1 or has_code_request:
        return "medium"
    return "low"


# Budget ranges by complexity (in tokens)
_BUDGETS = {
    "low": 150,
    "medium": 400,
    "high": 800,
}


class TokenBudgetInjector:
    """Inject token budget hints into the system prompt."""

    def inject(
        self,
        system: Optional[Union[str, List[Dict[str, Any]]]],
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[Union[str, List[Dict[str, Any]]]], bool]:
        """Add token budget to system prompt.

        Returns (new_system, was_injected).
        """
        complexity = _estimate_complexity(messages)
        budget = _BUDGETS[complexity]
        budget_text = f"\n\n[Token Budget: ~{budget} tokens for this response]"

        if system is None:
            return budget_text.strip(), True

        if isinstance(system, str):
            return system + budget_text, True

        if isinstance(system, list):
            # Append to the last text block
            new_system = list(system)
            for i in range(len(new_system) - 1, -1, -1):
                block = new_system[i]
                if isinstance(block, dict) and block.get("type") == "text":
                    new_block = dict(block)
                    new_block["text"] = block["text"] + budget_text
                    new_system[i] = new_block
                    return new_system, True
            # No text block found, add one
            new_system.append({"type": "text", "text": budget_text.strip()})
            return new_system, True

        return system, False
