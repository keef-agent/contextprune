"""Tests for the TALE-grounded TokenBudgetInjector and complexity classifier.

Tests validate:
- Each ComplexityLevel is reachable with appropriate inputs
- Token Elasticity safety: budget never < 50
- Factual questions → SIMPLE, code generation → COMPLEX
- Budget scales with context size within each level
- Injection format: instruction appears at END of system prompt
- Empty/None system prompt gets created with just the instruction
- Feature score breakdown is correct
- The full inject() returns correct tuple structure
"""

from __future__ import annotations

import re
import pytest

from contextprune.budget import (
    ComplexityLevel,
    TokenBudgetInjector,
    calculate_budget,
    classify_complexity,
    format_budget_instruction,
    _score_task_type,
    _score_reasoning_chain,
    _score_context_density,
    _score_tool_depth,
    _score_history_depth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def injector():
    return TokenBudgetInjector()


# ---------------------------------------------------------------------------
# ComplexityLevel reachability
# ---------------------------------------------------------------------------

class TestComplexityLevelReachability:
    """Each level must be reachable via classify_complexity."""

    def test_simple_factual_question(self):
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        level, score, _ = classify_complexity(messages)
        assert level == ComplexityLevel.SIMPLE
        assert score <= 5

    def test_medium_comparison_task(self):
        # Comparison task with:
        #   f1=2 (compare + "in detail"), f2=2 (reasoning markers in assistant turn),
        #   f3=1 (system ~500+ tokens), f4=1 (1 tool), f5=0 → total=6 → MEDIUM
        messages = [
            {
                "role": "assistant",
                "content": (
                    "First, let me outline the key factors. "
                    "Then we can compare them. However, there are important trade-offs. "
                    "Therefore, the choice depends on the use case."
                ),
            },
            {"role": "user", "content": "Compare Python and JavaScript for web development in detail."},
        ]
        # ~2000+ chars → ~500+ tokens → context_density=1
        system = "You are a technical expert. " + ("x" * 2000)
        tools = [{"name": "documentation_search"}]
        level, score, _ = classify_complexity(messages, tools=tools, system=system)
        assert level == ComplexityLevel.MEDIUM
        assert 6 <= score <= 10

    def test_complex_code_generation(self):
        # Generation task with:
        #   f1=3 (write + step-by-step bonus), f2=2 (markers in history),
        #   f3=2 (big system ~1500+ tokens), f4=3 (6 tools + "search" intent),
        #   f5=1 (3 messages) → total = 11 → COMPLEX
        system = "You are a senior software engineer. " * 200  # ~7200 chars → 1800 tokens
        tools = [{"name": f"t{i}"} for i in range(6)]  # 6 tools → f4=2 base + intent bonus
        messages = [
            {
                "role": "user",
                "content": "I need a caching system for my application.",
            },
            {
                "role": "assistant",
                "content": (
                    "First, let me understand your requirements. "
                    "Then we can design the right solution. "
                    "However, there are several patterns to consider. "
                    "Therefore, LRU is often the best choice."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write a complete Python function to search and implement "
                    "a binary search tree with insert, delete, and traversal step by step."
                ),
            },
        ]
        level, score, _ = classify_complexity(messages, tools=tools, system=system)
        assert level == ComplexityLevel.COMPLEX
        assert 11 <= score <= 15

    def test_very_complex_multi_objective(self):
        # Multi-objective + large context + many tools + deep history:
        #   f1=4 (analyze+implement, multi-objective + detail bonus),
        #   f2=4 (10+ markers across history), f3=2 (big context),
        #   f4=3 (15 tools, 11-20 range), f5=4 (25+ messages) → total=17 → VERY_COMPLEX
        tools = [{"name": f"tool_{i}"} for i in range(15)]  # 15 tools → f4=3
        long_history = []
        for i in range(12):  # 24 messages
            long_history.append({"role": "user", "content": f"Question {i} about the system."})
            long_history.append({
                "role": "assistant",
                "content": f"First this. Then that. However consider X. Therefore Y. But also Z because of A.",
            })
        long_history.append({
            "role": "user",
            "content": (
                "Analyze the current architecture and implement the changes, "
                "then document them comprehensively step by step in detail."
            ),
        })
        system = "x" * 6000  # 6000 chars → 1500 tokens → f3=2 (plus message context)
        level, score, _ = classify_complexity(long_history, tools=tools, system=system)
        assert level == ComplexityLevel.VERY_COMPLEX
        assert score >= 16


# ---------------------------------------------------------------------------
# Token Elasticity safety
# ---------------------------------------------------------------------------

class TestTokenElasticitySafety:
    """Budget must never drop below 50 tokens."""

    def test_budget_never_below_50(self):
        for level in ComplexityLevel:
            for ctx in [0, 1, 10, 50, 100]:
                b = calculate_budget(level, ctx)
                assert b >= 50, f"Budget {b} < 50 for level={level.value}, ctx={ctx}"

    def test_simple_min_budget(self):
        # Even at 0 context tokens, SIMPLE must give >= 50
        b = calculate_budget(ComplexityLevel.SIMPLE, 0)
        assert b >= 50

    def test_budget_floor_is_50(self):
        # Manually verify the floor: SIMPLE at 0 context gives max(50, 75) = 75
        b = calculate_budget(ComplexityLevel.SIMPLE, 0)
        assert b == 75  # low end of SIMPLE range


# ---------------------------------------------------------------------------
# Task type classification
# ---------------------------------------------------------------------------

class TestTaskTypeClassification:
    def test_factual_question_simple(self):
        messages = [{"role": "user", "content": "What is the boiling point of water?"}]
        level, _, _ = classify_complexity(messages)
        assert level == ComplexityLevel.SIMPLE

    def test_who_is_simple(self):
        messages = [{"role": "user", "content": "Who is the CEO of Apple?"}]
        level, _, _ = classify_complexity(messages)
        assert level == ComplexityLevel.SIMPLE

    def test_code_generation_complex(self):
        # With substantial context + tools + history, code generation → COMPLEX
        messages = [
            {"role": "user", "content": "I need to build a caching system."},
            {
                "role": "assistant",
                "content": (
                    "First, let me understand the requirements. "
                    "Then I can design a solution. "
                    "However, we need to consider memory constraints. "
                    "Therefore, LRU would be ideal."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write a comprehensive, step by step implementation of a LRU cache "
                    "in Python with all edge cases and tests."
                ),
            },
        ]
        system = "You are a Python expert with 10 years of experience. " * 120  # ~6480 chars → 1620 tokens
        tools = [
            {"name": "python_repl"}, {"name": "file_writer"},
            {"name": "test_runner"}, {"name": "code_search"}, {"name": "docs"},
        ]
        level, _, _ = classify_complexity(messages, tools=tools, system=system)
        assert level in (ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX)

    def test_implement_keyword_pushes_up(self):
        messages = [{"role": "user", "content": "Implement a REST API endpoint for user authentication."}]
        level, score, features = classify_complexity(messages)
        # task_type should be at least 2 (generation)
        assert features["task_type"] >= 2

    def test_step_by_step_bonus(self):
        msg_without = "Write a sorting algorithm."
        msg_with = "Write a sorting algorithm step by step."
        score_without = _score_task_type(msg_without)
        score_with = _score_task_type(msg_with)
        assert score_with > score_without

    def test_comprehensive_bonus(self):
        msg_without = "Explain recursion."
        msg_with = "Explain recursion in a comprehensive way."
        s1 = _score_task_type(msg_without)
        s2 = _score_task_type(msg_with)
        assert s2 > s1


# ---------------------------------------------------------------------------
# Budget scaling with context size
# ---------------------------------------------------------------------------

class TestBudgetScaling:
    """Budget should scale upward within each level as context grows."""

    def test_simple_scales_with_context(self):
        b_low = calculate_budget(ComplexityLevel.SIMPLE, 0)
        b_high = calculate_budget(ComplexityLevel.SIMPLE, 4000)
        assert b_high > b_low
        assert b_low >= 75
        assert b_high <= 150

    def test_medium_scales_with_context(self):
        b_low = calculate_budget(ComplexityLevel.MEDIUM, 0)
        b_high = calculate_budget(ComplexityLevel.MEDIUM, 4000)
        assert b_high > b_low
        assert b_low >= 200
        assert b_high <= 400

    def test_complex_scales_with_context(self):
        b_low = calculate_budget(ComplexityLevel.COMPLEX, 0)
        b_high = calculate_budget(ComplexityLevel.COMPLEX, 4000)
        assert b_high > b_low
        assert b_low >= 500
        assert b_high <= 800

    def test_very_complex_scales_with_context(self):
        b_low = calculate_budget(ComplexityLevel.VERY_COMPLEX, 0)
        b_high = calculate_budget(ComplexityLevel.VERY_COMPLEX, 4000)
        assert b_high > b_low
        assert b_low >= 900
        assert b_high <= 2000

    def test_context_factor_caps_at_1(self):
        # Context tokens >> 4000 should not exceed the high end of range
        b = calculate_budget(ComplexityLevel.SIMPLE, 100000)
        assert b == 150


# ---------------------------------------------------------------------------
# Injection format
# ---------------------------------------------------------------------------

class TestInjectionFormat:
    def test_instruction_at_end_of_system_prompt(self, injector):
        system = "You are a helpful assistant."
        messages = [{"role": "user", "content": "What is 2+2?"}]
        new_system, stats = injector.inject(messages, system)
        assert stats["injected"] is True
        # Instruction must be AFTER the original system content
        assert new_system.startswith("You are a helpful assistant.")
        assert new_system.index("You are a helpful assistant.") < new_system.rindex(stats["instruction"])

    def test_instruction_appended_not_prepended(self, injector):
        system = "ORIGINAL_SYSTEM_CONTENT"
        messages = [{"role": "user", "content": "Hello"}]
        new_system, stats = injector.inject(messages, system)
        # Original content comes first
        assert new_system.startswith("ORIGINAL_SYSTEM_CONTENT")
        # Instruction is at the end
        assert new_system.endswith(stats["instruction"])

    def test_simple_instruction_format(self):
        instr = format_budget_instruction(100, ComplexityLevel.SIMPLE)
        assert "100 tokens" in instr
        assert "concisely" in instr.lower() or "target" in instr.lower()

    def test_medium_instruction_format(self):
        instr = format_budget_instruction(300, ComplexityLevel.MEDIUM)
        assert "300 tokens" in instr
        assert "focused" in instr.lower() or "approximately" in instr.lower()

    def test_complex_instruction_format(self):
        instr = format_budget_instruction(600, ComplexityLevel.COMPLEX)
        assert "600 tokens" in instr
        assert "reasoning" in instr.lower() or "thorough" in instr.lower()

    def test_very_complex_instruction_format(self):
        instr = format_budget_instruction(1200, ComplexityLevel.VERY_COMPLEX)
        assert "1200 tokens" in instr
        assert "comprehensive" in instr.lower()


# ---------------------------------------------------------------------------
# None/empty system prompt
# ---------------------------------------------------------------------------

class TestNoneSystemPrompt:
    def test_none_system_creates_instruction(self, injector):
        messages = [{"role": "user", "content": "What is Python?"}]
        new_system, stats = injector.inject(messages, None)
        assert stats["injected"] is True
        assert new_system is not None
        assert len(new_system) > 0
        assert stats["instruction"] in new_system

    def test_empty_string_system(self, injector):
        messages = [{"role": "user", "content": "What is Python?"}]
        new_system, stats = injector.inject(messages, "")
        assert stats["injected"] is True
        assert stats["instruction"] in new_system

    def test_list_system_gets_instruction(self, injector):
        system = [{"type": "text", "text": "You are an assistant."}]
        messages = [{"role": "user", "content": "Hello"}]
        new_system, stats = injector.inject(messages, system)
        assert stats["injected"] is True
        assert isinstance(new_system, list)
        # Instruction should be in the last text block
        last_text = ""
        for block in reversed(new_system):
            if isinstance(block, dict) and block.get("type") == "text":
                last_text = block["text"]
                break
        assert stats["instruction"] in last_text

    def test_list_system_no_text_block(self, injector):
        system = [{"type": "image", "source": "..."}]
        messages = [{"role": "user", "content": "Hello"}]
        new_system, stats = injector.inject(messages, system)
        assert stats["injected"] is True
        assert any(
            isinstance(b, dict) and stats["instruction"] in b.get("text", "")
            for b in new_system
        )


# ---------------------------------------------------------------------------
# Feature score breakdown
# ---------------------------------------------------------------------------

class TestFeatureScoreBreakdown:
    def test_feature_scores_keys_present(self):
        messages = [{"role": "user", "content": "Hello"}]
        _, _, features = classify_complexity(messages)
        assert set(features.keys()) == {
            "task_type", "reasoning_chain", "context_density",
            "tool_depth", "history_depth",
        }

    def test_all_scores_in_range_0_to_4(self):
        messages = [
            {"role": "user", "content": "Implement a distributed hash table step by step"},
        ]
        tools = [{"name": f"t{i}"} for i in range(25)]
        _, _, features = classify_complexity(messages, tools=tools)
        for key, val in features.items():
            assert 0 <= val <= 4, f"Feature {key}={val} out of range"

    def test_task_type_factual_is_0(self):
        score = _score_task_type("What is the capital of France?")
        assert score == 0

    def test_task_type_generation_is_2_plus(self):
        score = _score_task_type("Write a function to reverse a string.")
        assert score >= 2

    def test_reasoning_chain_empty_is_0(self):
        score = _score_reasoning_chain("What is 2+2?", "")
        assert score == 0

    def test_reasoning_chain_many_markers(self):
        text = "first step then because therefore however but finally step step"
        score = _score_reasoning_chain(text, "")
        assert score >= 3

    def test_context_density_small_is_0(self):
        messages = [{"role": "user", "content": "Hi"}]
        score = _score_context_density(messages, None)
        assert score == 0

    def test_context_density_large_is_high(self):
        # ~7500 tokens worth of text (30000 chars)
        big_msg = [{"role": "user", "content": "a" * 30000}]
        score = _score_context_density(big_msg, None)
        assert score == 4

    def test_tool_depth_no_tools_is_0(self):
        score = _score_tool_depth(None, "hello")
        assert score == 0

    def test_tool_depth_many_tools_is_high(self):
        tools = [{"name": f"t{i}"} for i in range(25)]
        score = _score_tool_depth(tools, "hello")
        assert score == 4

    def test_tool_depth_intent_bonus(self):
        tools = [{"name": "search"}]
        score_plain = _score_tool_depth(tools, "tell me something")
        score_intent = _score_tool_depth(tools, "search for the latest news")
        assert score_intent > score_plain

    def test_history_depth_few_messages_is_0(self):
        messages = [{"role": "user", "content": "Hi"}]
        score = _score_history_depth(messages)
        assert score == 0

    def test_history_depth_many_messages_is_high(self):
        messages = [{"role": "user", "content": "m"} for _ in range(25)]
        score = _score_history_depth(messages)
        assert score == 4


# ---------------------------------------------------------------------------
# inject() return structure
# ---------------------------------------------------------------------------

class TestInjectReturnStructure:
    def test_returns_tuple_of_two(self, injector):
        messages = [{"role": "user", "content": "Hello"}]
        result = injector.inject(messages, "You are helpful.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_stats_dict_has_all_keys(self, injector):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        _, stats = injector.inject(messages, "You are helpful.")
        required_keys = {
            "complexity_level", "complexity_score", "feature_scores",
            "budget_tokens", "injected", "instruction",
        }
        assert required_keys.issubset(set(stats.keys()))

    def test_stats_complexity_level_is_valid_enum_value(self, injector):
        messages = [{"role": "user", "content": "Write a function"}]
        _, stats = injector.inject(messages, "You are helpful.")
        valid_values = {l.value for l in ComplexityLevel}
        assert stats["complexity_level"] in valid_values

    def test_stats_budget_tokens_positive(self, injector):
        messages = [{"role": "user", "content": "What is Python?"}]
        _, stats = injector.inject(messages, "You are helpful.")
        assert stats["budget_tokens"] > 0

    def test_stats_feature_scores_dict(self, injector):
        messages = [{"role": "user", "content": "Explain recursion"}]
        _, stats = injector.inject(messages, "You are helpful.")
        assert isinstance(stats["feature_scores"], dict)
        assert len(stats["feature_scores"]) == 5

    def test_injected_true_for_normal_request(self, injector):
        messages = [{"role": "user", "content": "Hello, what can you do?"}]
        _, stats = injector.inject(messages, "You are helpful.")
        assert stats["injected"] is True

    def test_disabled_injector_returns_unmodified(self):
        inj = TokenBudgetInjector(enabled=False)
        system = "You are helpful."
        messages = [{"role": "user", "content": "Hello"}]
        new_system, stats = inj.inject(messages, system)
        assert new_system == system
        assert stats["injected"] is False

    def test_instruction_in_stats_matches_injected_text(self, injector):
        system = "Base system."
        messages = [{"role": "user", "content": "What is Python?"}]
        new_system, stats = injector.inject(messages, system)
        assert stats["instruction"] in new_system

    def test_complexity_score_sums_feature_scores(self, injector):
        messages = [{"role": "user", "content": "Implement a binary search tree."}]
        _, stats = injector.inject(messages, "You are a coder.")
        feature_sum = sum(stats["feature_scores"].values())
        assert stats["complexity_score"] == feature_sum
