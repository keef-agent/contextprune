"""Token budget injection — TALE-grounded complexity classifier.

Implements the key findings from:
    Han et al. (2024). TALE: Token Budget Aware LLM Reasoning.
    arXiv:2412.18547. https://arxiv.org/abs/2412.18547

Core insight: explicit calibrated budgets ("Respond in N tokens") outperform
vague brevity instructions ("be concise"). But the budget must be *achievable*:
too-low budgets trigger Token Elasticity — the model uses MORE tokens than if
given no instruction at all. The classifier estimates complexity to select the
right budget range, then injects the instruction at the END of the system prompt.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .tokenizer import count_tokens


# ---------------------------------------------------------------------------
# Complexity levels
# ---------------------------------------------------------------------------

class ComplexityLevel(Enum):
    SIMPLE = "simple"           # factual recall, yes/no, single lookup
    MEDIUM = "medium"           # multi-step reasoning, summarization, comparison
    COMPLEX = "complex"         # synthesis, code generation, multi-hop reasoning
    VERY_COMPLEX = "very_complex"  # long-form generation, deep analysis, planning


# ---------------------------------------------------------------------------
# Feature scoring helpers
# ---------------------------------------------------------------------------

def _get_last_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract the text content of the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        parts.append(block["text"])
                return " ".join(parts)
    return ""


def _get_recent_assistant_text(messages: List[Dict[str, Any]], n: int = 3) -> str:
    """Extract text from up to the last n assistant messages."""
    texts = []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        texts.append(block["text"])
            if len(texts) >= n:
                break
    return " ".join(texts)


def _score_task_type(last_user: str) -> int:
    """Feature 1: task type (0–4).

    0 = factual recall, 1 = comparison/explanation, 2 = generation,
    3 = multi-objective. +1 bonus for detailed/comprehensive signals, cap at 4.
    """
    text = last_user.lower()

    # Multi-objective: multiple action verbs separated by "and"
    multi_obj_patterns = [
        r"\banalyze\b.{0,50}\band\b.{0,50}\b(recommend|implement|create|write|design)\b",
        r"\b(implement|create|write|design)\b.{0,50}\band\b.{0,50}\b(explain|document|test)\b",
        r"\b(explain|describe)\b.{0,50}\band\b.{0,50}\b(implement|create|build|write)\b",
    ]
    if any(re.search(p, text) for p in multi_obj_patterns):
        base = 3
    elif re.search(
        r"\b(write|create|implement|build|generate|draft|code|develop|design)\b", text
    ):
        base = 2
    elif re.search(
        r"\b(compare|contrast|explain|summarize|list|describe|outline|overview)\b", text
    ):
        base = 1
    elif re.search(r"\b(what is|what are|when did|who is|who was|where is|how many|define)\b", text):
        base = 0
    else:
        # Default: treat as medium signal if no clear factual markers
        base = 1

    # Bonus for depth/detail signals
    bonus = 0
    if re.search(r"\b(step by step|in detail|comprehensive|detailed|thorough|complete)\b", text):
        bonus = 1

    return min(4, base + bonus)


def _score_reasoning_chain(last_user: str, recent_assistant: str) -> int:
    """Feature 2: reasoning chain estimate (0–4).

    Count reasoning markers across last user message + recent assistant turns.
    """
    combined = (last_user + " " + recent_assistant).lower()
    markers = [
        "step", "first", "then", "finally", "because", "therefore",
        "however", "but", "next", "subsequently", "thus", "hence",
    ]
    count = sum(len(re.findall(r"\b" + m + r"\b", combined)) for m in markers)

    if count == 0:
        return 0
    elif count <= 2:
        return 1
    elif count <= 5:
        return 2
    elif count <= 9:
        return 3
    else:
        return 4


def _score_context_density(messages: List[Dict[str, Any]], system: Optional[str]) -> int:
    """Feature 3: context density (0–4).

    Estimate total tokens across all messages + system prompt.
    """
    total_chars = 0

    # System prompt
    if system:
        total_chars += len(system)

    # All messages
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("text"):
                    total_chars += len(block["text"])

    approx_tokens = total_chars // 4

    if approx_tokens < 500:
        return 0
    elif approx_tokens < 1500:
        return 1
    elif approx_tokens < 3000:
        return 2
    elif approx_tokens < 6000:
        return 3
    else:
        return 4


def _score_tool_depth(tools: Optional[List[Dict[str, Any]]], last_user: str) -> int:
    """Feature 4: tool depth (0–4).

    Score based on number of available tools + tool-invocation intent signals.
    """
    n_tools = len(tools) if tools else 0

    if n_tools == 0:
        base = 0
    elif n_tools <= 3:
        base = 1
    elif n_tools <= 10:
        base = 2
    elif n_tools <= 20:
        base = 3
    else:
        base = 4

    # Tool invocation intent in last user message
    text = last_user.lower()
    intent_signals = ["use the", "call", "run", "fetch", "search", "invoke", "execute"]
    bonus = 1 if any(sig in text for sig in intent_signals) else 0

    return min(4, base + bonus)


def _score_history_depth(messages: List[Dict[str, Any]]) -> int:
    """Feature 5: history depth (0–4).

    Score based on total number of messages in the conversation.
    """
    n = len(messages)
    if n <= 2:
        return 0
    elif n <= 5:
        return 1
    elif n <= 10:
        return 2
    elif n <= 20:
        return 3
    else:
        return 4


# ---------------------------------------------------------------------------
# Public classifier interface
# ---------------------------------------------------------------------------

def classify_complexity(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    system: Optional[str] = None,
) -> Tuple[ComplexityLevel, int, Dict[str, int]]:
    """Classify request complexity using TALE's 5-feature scoring scheme.

    Scores 5 features (0–4 each, total 0–20) then maps to a ComplexityLevel:
        0–5  → SIMPLE
        6–10 → MEDIUM
        11–15 → COMPLEX
        16–20 → VERY_COMPLEX

    Returns:
        (level, total_score, feature_scores)
        feature_scores keys: task_type, reasoning_chain, context_density,
                             tool_depth, history_depth
    """
    last_user = _get_last_user_text(messages)
    recent_asst = _get_recent_assistant_text(messages)

    f1 = _score_task_type(last_user)
    f2 = _score_reasoning_chain(last_user, recent_asst)
    f3 = _score_context_density(messages, system)
    f4 = _score_tool_depth(tools, last_user)
    f5 = _score_history_depth(messages)

    total = f1 + f2 + f3 + f4 + f5
    feature_scores = {
        "task_type": f1,
        "reasoning_chain": f2,
        "context_density": f3,
        "tool_depth": f4,
        "history_depth": f5,
    }

    if total <= 5:
        level = ComplexityLevel.SIMPLE
    elif total <= 10:
        level = ComplexityLevel.MEDIUM
    elif total <= 15:
        level = ComplexityLevel.COMPLEX
    else:
        level = ComplexityLevel.VERY_COMPLEX

    return level, total, feature_scores


# ---------------------------------------------------------------------------
# Budget calculation
# ---------------------------------------------------------------------------

def calculate_budget(level: ComplexityLevel, context_tokens: int) -> int:
    """Calculate optimal token budget following TALE's calibration principle.

    Key insight from TALE: budget must be achievable. If budget < 0.3 * expected_output,
    Token Elasticity causes MORE tokens to be used.

    Scales within the level's range based on context size: more context generally
    means more to synthesize, so we bias toward the higher end of the range.

    Returns the recommended output token budget as an integer.
    """
    base_ranges = {
        ComplexityLevel.SIMPLE: (75, 150),
        ComplexityLevel.MEDIUM: (200, 400),
        ComplexityLevel.COMPLEX: (500, 800),
        ComplexityLevel.VERY_COMPLEX: (900, 2000),
    }

    low, high = base_ranges[level]

    # Scale within range based on context size
    context_factor = min(1.0, context_tokens / 4000)
    budget = int(low + (high - low) * context_factor)

    # Token Elasticity safety: never go below 50
    return max(50, budget)


# ---------------------------------------------------------------------------
# Injection format
# ---------------------------------------------------------------------------

def format_budget_instruction(budget: int, level: ComplexityLevel) -> str:
    """Format the token budget instruction for injection into system prompt.

    TALE found that explicit, calibrated phrasing beats vague brevity
    instructions. The format is tuned per complexity level.
    """
    if level == ComplexityLevel.SIMPLE:
        return f"Respond concisely. Target: {budget} tokens or fewer."

    elif level == ComplexityLevel.MEDIUM:
        return f"Provide a focused response. Target length: approximately {budget} tokens."

    elif level == ComplexityLevel.COMPLEX:
        return (
            f"This task requires careful reasoning. "
            f"Aim for a complete response in approximately {budget} tokens. "
            f"Be thorough but avoid unnecessary repetition."
        )

    elif level == ComplexityLevel.VERY_COMPLEX:
        return (
            f"This is a complex task. Provide a comprehensive response, "
            f"targeting approximately {budget} tokens. "
            f"Structure your response clearly and cover all required aspects."
        )

    # Fallback (shouldn't be reached)
    return f"Target length: approximately {budget} tokens."


# ---------------------------------------------------------------------------
# TokenBudgetInjector
# ---------------------------------------------------------------------------

class TokenBudgetInjector:
    """TALE-grounded token budget injection for LLM requests.

    Implements the key findings from:
    Han et al. (2024). TALE: Token Budget Aware LLM Reasoning.
    arXiv:2412.18547. https://arxiv.org/abs/2412.18547

    Key principle: explicit calibrated budgets outperform instruction-based
    brevity ("be concise"). Budget must be achievable to avoid Token Elasticity
    (too-small budgets cause MORE tokens, not fewer).

    The budget instruction is injected at the END of the system prompt — models
    weight end-of-system-prompt instructions more heavily than leading text.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_context_tokens: int = 200,  # don't inject for tiny requests
        safety_margin: float = 0.3,     # never budget below 30% of estimated need
    ) -> None:
        self.enabled = enabled
        self.min_context_tokens = min_context_tokens
        self.safety_margin = safety_margin

    def inject(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[Union[str, List[Dict[str, Any]]]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Union[str, List[Dict[str, Any]]]], Dict[str, Any]]:
        """Analyze request complexity and inject token budget into system prompt.

        The budget instruction is appended to the END of the system prompt so
        that model attention naturally weights it highest.

        Returns:
            (modified_system, stats)
            stats = {
                "complexity_level": str,
                "complexity_score": int,
                "feature_scores": dict,   # breakdown of 5 features
                "budget_tokens": int,
                "injected": bool,
                "instruction": str,
            }
        """
        # Build baseline stats
        stats: Dict[str, Any] = {
            "complexity_level": ComplexityLevel.SIMPLE.value,
            "complexity_score": 0,
            "feature_scores": {},
            "budget_tokens": 0,
            "injected": False,
            "instruction": "",
        }

        if not self.enabled:
            return system, stats

        # Extract plain system string for classifier (handles list format)
        system_str: Optional[str] = None
        if isinstance(system, str):
            system_str = system
        elif isinstance(system, list):
            parts = []
            for block in system:
                if isinstance(block, dict) and block.get("text"):
                    parts.append(block["text"])
            system_str = " ".join(parts) if parts else None

        # Estimate total context tokens to gate on min_context_tokens
        total_chars = len(system_str) if system_str else 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        total_chars += len(block["text"])
        context_tokens = max(1, total_chars // 4)

        # Skip injection for trivially small requests
        if context_tokens < self.min_context_tokens and not messages:
            return system, stats

        # Classify complexity
        level, score, feature_scores = classify_complexity(messages, tools, system_str)

        # Calculate budget
        budget = calculate_budget(level, context_tokens)

        # Format instruction
        instruction = format_budget_instruction(budget, level)

        # Populate stats
        stats["complexity_level"] = level.value
        stats["complexity_score"] = score
        stats["feature_scores"] = feature_scores
        stats["budget_tokens"] = budget
        stats["instruction"] = instruction

        # Inject at the END of the system prompt
        budget_suffix = f"\n\n{instruction}"

        if system is None:
            new_system: Optional[Union[str, List[Dict[str, Any]]]] = instruction
            stats["injected"] = True
            return new_system, stats

        if isinstance(system, str):
            stats["injected"] = True
            return system + budget_suffix, stats

        if isinstance(system, list):
            new_system_list = list(system)
            # Append to the last text block
            for i in range(len(new_system_list) - 1, -1, -1):
                block = new_system_list[i]
                if isinstance(block, dict) and block.get("type") == "text":
                    new_block = dict(block)
                    new_block["text"] = block["text"] + budget_suffix
                    new_system_list[i] = new_block
                    stats["injected"] = True
                    return new_system_list, stats
            # No text block found — append a new one
            new_system_list.append({"type": "text", "text": instruction})
            stats["injected"] = True
            return new_system_list, stats

        # Unrecognised system format — return unchanged
        return system, stats
