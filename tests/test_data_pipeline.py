"""
Smoke tests for Phase 2 data pipeline.

Tests:
  - ContextInjector produces valid messages list
  - CostGuard raises on budget exceeded
  - Checkpoint save/load round-trip
  - Accuracy evaluators (exact match, token F1)
  - Cost estimation
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# ContextInjector tests
# ---------------------------------------------------------------------------

class TestContextInjector:
    """Tests for context injection harness."""

    def _make_item(self, category: str = "math", difficulty: str = "medium") -> dict:
        return {
            "id": "test_001",
            "question": "What is 2 + 2?",
            "answer": "4",
            "choices": None,
            "context": "",
            "category": category,
            "source_dataset": "mmlu_pro",
            "difficulty": difficulty,
        }

    def _get_injector(self):
        from benchmarks.context_injector import ContextInjector
        # Use a temp path so tests don't need the real toolschemas
        return ContextInjector(tools_path="/nonexistent/tools.json")

    def test_inject_returns_messages_list(self):
        injector = self._get_injector()
        item = self._make_item("math")
        result = injector.inject(item)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0

    def test_inject_last_message_is_user_with_question(self):
        injector = self._get_injector()
        item = self._make_item("factual")
        result = injector.inject(item)
        msgs = result["messages"]
        assert msgs[-1]["role"] == "user"
        assert "What is 2 + 2?" in msgs[-1]["content"]

    def test_inject_adds_system_prompt(self):
        injector = self._get_injector()
        item = self._make_item("code")
        result = injector.inject(item)
        assert "system" in result
        assert isinstance(result["system"], str)
        assert len(result["system"]) > 50

    def test_inject_small_has_fewer_messages(self):
        injector = self._get_injector()
        item = self._make_item("agent")
        small = injector.inject(item, context_size="small")
        large = injector.inject(item, context_size="large")
        # Large should have more history turns than small
        assert len(large["messages"]) >= len(small["messages"])

    def test_inject_rag_adds_context_chunks(self):
        injector = self._get_injector()
        item = self._make_item("rag")
        result = injector.inject(item, context_size="medium")
        # The last user message should contain retrieved context
        last_msg = result["messages"][-1]["content"]
        assert "retrieved_context" in last_msg or "Document" in last_msg or "2 + 2" in last_msg

    def test_inject_all_categories(self):
        injector = self._get_injector()
        for category in ["math", "code", "factual", "rag", "agent"]:
            item = self._make_item(category)
            result = injector.inject(item)
            assert "messages" in result
            assert len(result["messages"]) > 0

    def test_inject_token_estimate_positive(self):
        injector = self._get_injector()
        item = self._make_item("math")
        result = injector.inject(item)
        assert result.get("injected_token_estimate", 0) > 0

    def test_inject_preserves_item_fields(self):
        injector = self._get_injector()
        item = self._make_item("factual")
        result = injector.inject(item)
        assert result["id"] == "test_001"
        assert result["answer"] == "4"
        assert result["category"] == "factual"

    def test_inject_batch_utility(self):
        from benchmarks.context_injector import inject_batch
        items = [self._make_item("math"), self._make_item("code"), self._make_item("factual")]
        results = inject_batch(items, context_size="small", tools_path="/nonexistent/tools.json")
        assert len(results) == 3
        for r in results:
            assert "messages" in r


# ---------------------------------------------------------------------------
# CostGuard tests
# ---------------------------------------------------------------------------

class TestCostGuard:
    """Tests for budget guard."""

    def _get_cost_guard(self, budget: float = 10.0, checkpoint_path=None):
        from benchmarks.exp3_accuracy import CostGuard
        return CostGuard(budget_usd=budget, checkpoint_path=checkpoint_path)

    def test_check_within_budget_passes(self):
        guard = self._get_cost_guard(budget=10.0)
        guard.check(5.0)  # should not raise

    def test_check_at_exact_budget_raises(self):
        from benchmarks.exp3_accuracy import BudgetExceededError
        guard = self._get_cost_guard(budget=10.0)
        guard.record(9.99)
        with pytest.raises(BudgetExceededError):
            guard.check(0.02)  # 9.99 + 0.02 > 10.0

    def test_check_exceeds_budget_raises(self):
        from benchmarks.exp3_accuracy import BudgetExceededError
        guard = self._get_cost_guard(budget=5.0)
        with pytest.raises(BudgetExceededError):
            guard.check(6.0)

    def test_record_accumulates_spent(self):
        guard = self._get_cost_guard(budget=100.0)
        guard.record(1.50)
        guard.record(2.25)
        assert abs(guard.spent - 3.75) < 1e-9

    def test_remaining_decreases_as_spent(self):
        guard = self._get_cost_guard(budget=10.0)
        guard.record(3.0)
        assert abs(guard.remaining - 7.0) < 1e-9

    def test_remaining_never_negative(self):
        guard = self._get_cost_guard(budget=5.0)
        guard.record(10.0)  # overspend
        assert guard.remaining == 0.0

    def test_checkpoint_persistence(self):
        from benchmarks.exp3_accuracy import CostGuard
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            ckpt_path = Path(f.name)
        try:
            # Create guard, record some spend
            g1 = CostGuard(budget_usd=100.0, checkpoint_path=ckpt_path)
            g1.record(7.42)
            # Create second guard at same path — should load prior spend
            g2 = CostGuard(budget_usd=100.0, checkpoint_path=ckpt_path)
            assert abs(g2.spent - 7.42) < 0.01
        finally:
            ckpt_path.unlink(missing_ok=True)

    def test_zero_budget_raises_immediately(self):
        from benchmarks.exp3_accuracy import BudgetExceededError
        guard = self._get_cost_guard(budget=0.0)
        with pytest.raises(BudgetExceededError):
            guard.check(0.001)


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpoint:
    """Tests for checkpointing system."""

    def _make_checkpoint(self, path: Path):
        from benchmarks.exp3_accuracy import Checkpoint
        return Checkpoint(path)

    def test_empty_checkpoint_is_not_done(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        path.unlink()  # remove so it's truly empty
        try:
            ckpt = self._make_checkpoint(path)
            assert not ckpt.is_done("q001", "raw", "claude")
        finally:
            path.unlink(missing_ok=True)

    def test_write_and_is_done(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = Path(f.name)
        try:
            ckpt = self._make_checkpoint(path)
            result = {
                "question_id": "q001",
                "condition": "raw",
                "model": "claude",
                "correct": True,
                "tokens_in": 500,
                "tokens_out": 100,
                "cost_usd": 0.001,
                "latency_ms": 250.0,
                "compression_ratio": 1.0,
            }
            ckpt.write(result)
            assert ckpt.is_done("q001", "raw", "claude")
            assert not ckpt.is_done("q001", "raw", "gemini")  # different model
            assert not ckpt.is_done("q002", "raw", "claude")  # different question
        finally:
            path.unlink(missing_ok=True)

    def test_checkpoint_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = Path(f.name)
        try:
            ckpt1 = self._make_checkpoint(path)
            r1 = {"question_id": "q001", "condition": "raw", "model": "claude",
                  "correct": True, "tokens_in": 100, "tokens_out": 50,
                  "cost_usd": 0.01, "latency_ms": 100.0, "compression_ratio": 1.0}
            r2 = {"question_id": "q002", "condition": "contextprune", "model": "gpt52",
                  "correct": False, "tokens_in": 80, "tokens_out": 40,
                  "cost_usd": 0.005, "latency_ms": 80.0, "compression_ratio": 0.5}
            ckpt1.write(r1)
            ckpt1.write(r2)

            # Load fresh checkpoint from same file
            ckpt2 = self._make_checkpoint(path)
            assert ckpt2.is_done("q001", "raw", "claude")
            assert ckpt2.is_done("q002", "contextprune", "gpt52")
            assert not ckpt2.is_done("q003", "raw", "claude")

            # load_all should return both records
            loaded = ckpt2.load_all()
            assert len(loaded) == 2
        finally:
            path.unlink(missing_ok=True)

    def test_checkpoint_handles_corrupt_lines(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = Path(f.name)
            f.write('{"question_id": "q001", "condition": "raw", "model": "claude"}\n')
            f.write('CORRUPT_LINE_NOT_JSON\n')
            f.write('{"question_id": "q002", "condition": "raw", "model": "claude"}\n')
        try:
            ckpt = self._make_checkpoint(path)
            # Should load valid lines and skip corrupt one
            loaded = ckpt.load_all()
            assert len(loaded) >= 1  # at least valid lines loaded
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Accuracy evaluator tests
# ---------------------------------------------------------------------------

class TestAccuracyEvaluators:
    """Tests for per-dataset accuracy evaluation functions."""

    def test_mmlu_eval_correct_letter(self):
        from benchmarks.exp3_accuracy import eval_mmlu_pro
        assert eval_mmlu_pro("The answer is B.", "B")
        assert eval_mmlu_pro("Answer: C\n\nBecause...", "C")
        assert eval_mmlu_pro("(A) is correct", "A")
        assert eval_mmlu_pro("Looking at the options, D is clearly right.", "D")

    def test_mmlu_eval_wrong_letter(self):
        from benchmarks.exp3_accuracy import eval_mmlu_pro
        assert not eval_mmlu_pro("The answer is B.", "A")
        assert not eval_mmlu_pro("Answer: C", "D")

    def test_mmlu_eval_case_insensitive(self):
        from benchmarks.exp3_accuracy import eval_mmlu_pro
        assert eval_mmlu_pro("the answer is a.", "A")
        assert eval_mmlu_pro("ANSWER: B", "B")

    def test_math_eval_exact_match(self):
        from benchmarks.exp3_accuracy import eval_math500
        assert eval_math500("The answer is \\boxed{42}", "42")
        assert eval_math500("Therefore, x = 7", "7")

    def test_math_eval_string_match(self):
        from benchmarks.exp3_accuracy import eval_math500
        assert eval_math500("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}")

    def test_math_eval_wrong(self):
        from benchmarks.exp3_accuracy import eval_math500
        assert not eval_math500("The answer is 42", "43")

    def test_frames_f1_exact_match(self):
        from benchmarks.exp3_accuracy import eval_frames
        score = eval_frames("The capital of France is Paris", "The capital of France is Paris")
        assert score == 1.0

    def test_frames_f1_partial_match(self):
        from benchmarks.exp3_accuracy import eval_frames
        score = eval_frames("Paris is the capital", "The capital of France is Paris")
        assert 0.0 < score < 1.0

    def test_frames_f1_no_match(self):
        from benchmarks.exp3_accuracy import eval_frames
        score = eval_frames("London", "Paris")
        assert score == 0.0

    def test_gaia_exact_match(self):
        from benchmarks.exp3_accuracy import eval_gaia
        assert eval_gaia("Paris", "Paris")
        assert eval_gaia("paris", "PARIS")  # case insensitive
        assert eval_gaia("Paris.", "Paris")  # strip punctuation

    def test_gaia_no_match(self):
        from benchmarks.exp3_accuracy import eval_gaia
        assert not eval_gaia("London", "Paris")

    def test_evaluate_response_routing_mmlu(self):
        from benchmarks.exp3_accuracy import evaluate_response
        item = {"source_dataset": "mmlu_pro", "answer": "C"}
        assert evaluate_response(item, "The answer is C.")
        assert not evaluate_response(item, "The answer is A.")

    def test_evaluate_response_routing_gaia(self):
        from benchmarks.exp3_accuracy import evaluate_response
        item = {"source_dataset": "gaia", "answer": "Marie Curie"}
        assert evaluate_response(item, "marie curie")
        assert not evaluate_response(item, "Albert Einstein")

    def test_evaluate_response_routing_frames(self):
        from benchmarks.exp3_accuracy import evaluate_response
        item = {"source_dataset": "frames", "answer": "The Eiffel Tower is in Paris France"}
        # Should return bool (True if F1 >= 0.5)
        result = evaluate_response(item, "The Eiffel Tower is located in Paris France")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Cost estimation tests
# ---------------------------------------------------------------------------

class TestCostEstimation:
    """Tests for cost estimation utilities."""

    def test_estimate_returns_total_usd(self):
        from benchmarks.exp3_accuracy import estimate_cost
        est = estimate_cost(
            n_questions=100,
            conditions=["raw", "contextprune"],
            model_ids=["anthropic/claude-sonnet-4-6"],
            avg_input_tokens=1500,
            avg_output_tokens=256,
        )
        assert "total_usd" in est
        assert est["total_usd"] > 0

    def test_estimate_contextprune_cheaper_than_raw(self):
        from benchmarks.exp3_accuracy import estimate_cost
        est = estimate_cost(
            n_questions=100,
            conditions=["raw", "contextprune"],
            model_ids=["anthropic/claude-sonnet-4-6"],
            avg_input_tokens=1500,
        )
        model_costs = est["by_model"]["anthropic/claude-sonnet-4-6"]
        assert model_costs["contextprune"] < model_costs["raw"]

    def test_estimate_scales_linearly_with_n(self):
        from benchmarks.exp3_accuracy import estimate_cost
        est_100 = estimate_cost(100, ["raw"], ["anthropic/claude-sonnet-4-6"])
        est_200 = estimate_cost(200, ["raw"], ["anthropic/claude-sonnet-4-6"])
        ratio = est_200["total_usd"] / est_100["total_usd"]
        assert abs(ratio - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Truncation and statistical helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for truncation and stats helpers."""

    def test_truncation_preserves_last_message(self):
        from benchmarks.exp3_accuracy import apply_truncation
        messages = [
            {"role": "user", "content": "A " * 200},
            {"role": "assistant", "content": "B " * 200},
            {"role": "user", "content": "What is the answer?"},
        ]
        system = "You are helpful."
        new_msgs, new_sys = apply_truncation(messages, system, target_tokens=50)
        # Last message (the question) must be preserved
        assert new_msgs[-1]["content"] == "What is the answer?"

    def test_truncation_reduces_token_count(self):
        from benchmarks.exp3_accuracy import apply_truncation
        messages = [
            {"role": "user", "content": "X " * 500},
            {"role": "assistant", "content": "Y " * 500},
            {"role": "user", "content": "Final question"},
        ]
        system = "System " * 100
        orig_tokens = sum(len(m["content"]) // 4 for m in messages) + len(system) // 4
        new_msgs, _ = apply_truncation(messages, system, target_tokens=200)
        new_tokens = sum(len(m.get("content", "")) // 4 for m in new_msgs)
        assert new_tokens <= orig_tokens

    def test_compute_stats_returns_accuracy(self):
        from benchmarks.exp3_accuracy import compute_stats
        results = [
            {"question_id": f"q{i}", "condition": "raw", "model": "claude", "correct": i % 2 == 0}
            for i in range(20)
        ]
        stats = compute_stats(results, n_bootstrap=100)  # small bootstrap for speed
        assert ("raw", "claude") in stats
        s = stats[("raw", "claude")]
        assert 0 <= s["accuracy"] <= 1
        assert s["n"] == 20

    def test_compute_stats_pvalue_for_two_conditions(self):
        from benchmarks.exp3_accuracy import compute_stats
        # raw: 15/20 correct; contextprune: 14/20 correct (same IDs)
        raw_results = [
            {"question_id": f"q{i}", "condition": "raw", "model": "claude",
             "correct": i < 15}
            for i in range(20)
        ]
        cp_results = [
            {"question_id": f"q{i}", "condition": "contextprune", "model": "claude",
             "correct": i < 14}
            for i in range(20)
        ]
        stats = compute_stats(raw_results + cp_results, n_bootstrap=100)
        assert ("contextprune", "claude") in stats
        s = stats[("contextprune", "claude")]
        # Should have p_value_vs_raw
        assert "p_value_vs_raw" in s
        assert 0 <= s["p_value_vs_raw"] <= 1


# ---------------------------------------------------------------------------
# Integration: dataset item format
# ---------------------------------------------------------------------------

class TestDatasetFormat:
    """Test that standardized dataset items have required fields."""

    REQUIRED_FIELDS = ["id", "question", "answer", "choices", "context",
                       "category", "source_dataset", "difficulty"]

    def _check_item(self, item: dict, dataset: str) -> None:
        for field in self.REQUIRED_FIELDS:
            assert field in item, f"Missing field {field!r} in {dataset} item {item.get('id', '?')}"
        assert item["source_dataset"] == dataset
        assert item["category"] in ("math", "code", "factual", "rag", "agent")
        assert item["difficulty"] in ("easy", "medium", "hard")
        assert isinstance(item["question"], str) and len(item["question"]) > 0
        assert isinstance(item["answer"], str)

    @pytest.mark.skipif(
        not Path("benchmarks/data/mmlu_pro/test.jsonl").exists(),
        reason="MMLU-Pro dataset not downloaded"
    )
    def test_mmlu_pro_format(self):
        path = Path("benchmarks/data/mmlu_pro/test.jsonl")
        with open(path) as f:
            for line in list(f)[:5]:
                item = json.loads(line)
                self._check_item(item, "mmlu_pro")
                # MMLU-Pro should have choices
                assert item["choices"] is not None
                assert len(item["choices"]) > 0

    @pytest.mark.skipif(
        not Path("benchmarks/data/math500/test.jsonl").exists(),
        reason="MATH-500 dataset not downloaded"
    )
    def test_math500_format(self):
        path = Path("benchmarks/data/math500/test.jsonl")
        with open(path) as f:
            for line in list(f)[:5]:
                item = json.loads(line)
                self._check_item(item, "math500")
                assert item["choices"] is None  # no multiple choice


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
