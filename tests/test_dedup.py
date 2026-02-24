"""Tests for SemanticDeduplicator."""

from contextprune.dedup import SemanticDeduplicator


class TestSemanticDeduplicator:
    def setup_method(self):
        self.dedup = SemanticDeduplicator(similarity_threshold=0.85)

    def test_no_duplicates_unchanged(self):
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "Tell me about quantum computing."},
        ]
        new_msgs, new_sys, removed = self.dedup.deduplicate(messages)
        assert removed == 0
        assert len(new_msgs) == 3

    def test_exact_duplicate_removed(self):
        messages = [
            {"role": "user", "content": "The database connection uses PostgreSQL on port 5432."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "The database connection uses PostgreSQL on port 5432."},
        ]
        new_msgs, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 1

    def test_near_duplicate_removed(self):
        messages = [
            {"role": "user", "content": "The application uses PostgreSQL database on port 5432 for data storage."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "The application uses PostgreSQL database on port 5432 for storing data."},
        ]
        new_msgs, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 1

    def test_system_prompt_dedup(self):
        system = "You are a helpful coding assistant. You help users write Python code."
        messages = [
            {"role": "user", "content": "You are a helpful coding assistant. Help me write a function."},
        ]
        new_msgs, new_system, removed = self.dedup.deduplicate(messages, system=system)
        assert removed >= 1

    def test_preserves_message_structure(self):
        messages = [
            {"role": "user", "content": "Hello there, how are you today?"},
            {"role": "assistant", "content": "I am doing well, thanks for asking!"},
        ]
        new_msgs, _, _ = self.dedup.deduplicate(messages)
        assert all("role" in m for m in new_msgs)
        assert all("content" in m for m in new_msgs)

    def test_non_string_content_passthrough(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": "Hi there!"},
        ]
        new_msgs, _, _ = self.dedup.deduplicate(messages)
        assert len(new_msgs) == 2
        assert isinstance(new_msgs[0]["content"], list)

    def test_empty_messages(self):
        new_msgs, new_sys, removed = self.dedup.deduplicate([])
        assert new_msgs == []
        assert removed == 0

    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello world."}]
        new_msgs, _, removed = self.dedup.deduplicate(messages)
        assert len(new_msgs) == 1
        assert removed == 0

    def test_threshold_affects_sensitivity(self):
        """Lower threshold should catch more duplicates."""
        messages = [
            {"role": "user", "content": "The server runs on port 8080 with nginx."},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "The web server is nginx running on port 8080."},
        ]
        strict = SemanticDeduplicator(similarity_threshold=0.95)
        _, _, removed_strict = strict.deduplicate(messages)

        loose = SemanticDeduplicator(similarity_threshold=0.5)
        _, _, removed_loose = loose.deduplicate(messages)

        assert removed_loose >= removed_strict

    def test_many_duplicates(self):
        """Should handle many repeated sentences efficiently."""
        base = "The system requires Python 3.9 or higher to run properly."
        messages = [
            {"role": "user", "content": base},
        ]
        for i in range(10):
            messages.append({"role": "assistant", "content": f"Response {i}."})
            messages.append({"role": "user", "content": base + f" Additional note {i}."})

        _, _, removed = self.dedup.deduplicate(messages)
        assert removed >= 5
