"""Tests for TokenBudgetInjector."""

from contextprune.budget import TokenBudgetInjector, _estimate_complexity


class TestComplexityEstimation:
    def test_simple_message_low(self):
        messages = [{"role": "user", "content": "Hi"}]
        assert _estimate_complexity(messages) == "low"

    def test_code_request_medium(self):
        messages = [{"role": "user", "content": "Write a hello world function"}]
        assert _estimate_complexity(messages) == "medium"

    def test_complex_request_high(self):
        messages = [
            {
                "role": "user",
                "content": (
                    "Please provide a comprehensive, detailed, step by step "
                    "explanation of how the Linux kernel handles memory "
                    "management, including virtual memory, paging, and the "
                    "slab allocator. Compare and analyze the different "
                    "approaches used across kernel versions."
                ),
            }
        ]
        assert _estimate_complexity(messages) == "high"

    def test_empty_messages(self):
        assert _estimate_complexity([]) == "low"

    def test_no_user_message(self):
        messages = [{"role": "assistant", "content": "Hello!"}]
        assert _estimate_complexity(messages) == "low"


class TestTokenBudgetInjector:
    def setup_method(self):
        self.injector = TokenBudgetInjector()

    def test_inject_into_string_system(self):
        system = "You are a helpful assistant."
        messages = [{"role": "user", "content": "Hi"}]
        new_system, injected = self.injector.inject(system, messages)
        assert injected is True
        assert "[Token Budget:" in new_system
        assert "~150 tokens" in new_system

    def test_inject_none_system(self):
        messages = [{"role": "user", "content": "Hi"}]
        new_system, injected = self.injector.inject(None, messages)
        assert injected is True
        assert "[Token Budget:" in new_system

    def test_inject_into_list_system(self):
        system = [{"type": "text", "text": "You are helpful."}]
        messages = [{"role": "user", "content": "Hi"}]
        new_system, injected = self.injector.inject(system, messages)
        assert injected is True
        assert isinstance(new_system, list)
        assert "[Token Budget:" in new_system[-1]["text"]

    def test_budget_scales_with_complexity(self):
        simple_msgs = [{"role": "user", "content": "Hi"}]
        complex_msgs = [
            {
                "role": "user",
                "content": "Explain in comprehensive detail how to implement and analyze a distributed system architecture step by step.",
            }
        ]

        simple_sys, _ = self.injector.inject("System.", simple_msgs)
        complex_sys, _ = self.injector.inject("System.", complex_msgs)

        # Extract budget numbers
        import re

        simple_match = re.search(r"~(\d+) tokens", simple_sys)
        complex_match = re.search(r"~(\d+) tokens", complex_sys)

        assert simple_match and complex_match
        assert int(complex_match.group(1)) > int(simple_match.group(1))

    def test_preserves_original_system(self):
        system = "You are a helpful assistant."
        messages = [{"role": "user", "content": "Hi"}]
        new_system, _ = self.injector.inject(system, messages)
        assert new_system.startswith("You are a helpful assistant.")

    def test_list_system_no_text_block(self):
        system = [{"type": "image", "source": "..."}]
        messages = [{"role": "user", "content": "Hi"}]
        new_system, injected = self.injector.inject(system, messages)
        assert injected is True
        assert any(
            isinstance(b, dict) and "[Token Budget:" in b.get("text", "")
            for b in new_system
        )
