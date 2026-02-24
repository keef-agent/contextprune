"""Tests for ToolSchemaFilter (embedding-based implementation).

Unit tests mock the embedding model so no model download is required.
Integration tests (marked @pytest.mark.integration) use real embeddings.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import numpy as np
import pytest

from contextprune.tool_filter import ToolSchemaFilter


# ---------------------------------------------------------------------------
# Shared embedding mock (mirrors test_dedup.py mock for consistency)
# ---------------------------------------------------------------------------

_KEYWORD_GROUPS = [
    (["postgresql", "database", "port 5432", "data storage", "storing data"], 0),
    (["weather", "forecast", "temperature", "climate"], 1),
    (["python", "code", "function", "programming", "coding"], 2),
    (["file", "read", "write", "directory", "config"], 3),
    (["sql", "query"], 4),
    (["web", "search", "internet", "browse"], 5),
    (["email", "send", "message"], 6),
    (["helpful", "assistant"], 7),
    (["nginx", "server", "port 8080", "web server"], 8),
    (["quantum", "computing"], 9),
    (["stock", "price", "financial", "market", "ticker", "financial_data", "fetch"], 10),
    (["deploy", "service", "production", "deployment"], 11),
    (["git", "commit", "version"], 12),
    (["image", "resize", "photo", "picture"], 13),
    (["slack", "channel", "notification"], 14),
    (["translate", "language", "text"], 15),
    (["compress", "archive", "zip", "gzip"], 16),
    (["task", "project", "tracker", "create"], 17),
]

_DIM = 24
_embed_cache: dict = {}
_unique_counter = [len(_KEYWORD_GROUPS)]


def _mock_embed(texts, model_name, prefix=""):
    """Content-aware embedding mock for unit tests."""
    results = []
    for text in texts:
        clean = re.sub(r"^(search_document: |search_query: )", "", text).lower()

        if clean not in _embed_cache:
            best_group = None
            best_count = 0
            for keywords, group_idx in _KEYWORD_GROUPS:
                count = sum(1 for kw in keywords if kw in clean)
                if count > best_count:
                    best_count = count
                    best_group = group_idx

            vec = np.zeros(_DIM)
            if best_group is not None:
                vec[best_group] = 1.0
            else:
                slot = _unique_counter[0] % _DIM
                _unique_counter[0] += 1
                vec[slot] = 1.0

            _embed_cache[clean] = vec

        results.append(_embed_cache[clean])

    return np.array(results, dtype=np.float32)


def _mock_cosine_sim(a, b):
    return float(np.dot(a, b))


@pytest.fixture(autouse=True)
def patch_embeddings(request, monkeypatch):
    """Patch embedding module for unit tests. Skipped for @pytest.mark.integration tests."""
    if request.node.get_closest_marker("integration"):
        return  # Let integration tests use real embeddings
    monkeypatch.setattr("contextprune.embeddings.embed", _mock_embed)
    monkeypatch.setattr("contextprune.embeddings.cosine_similarity", _mock_cosine_sim)
    _embed_cache.clear()
    _unique_counter[0] = len(_KEYWORD_GROUPS)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_tool(name: str, description: str, params: dict = None) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": params or {},
        },
    }


SAMPLE_TOOLS = [
    _make_tool("get_weather", "Get the current weather for a location", {"location": {"type": "string"}}),
    _make_tool("search_web", "Search the web for information", {"query": {"type": "string"}}),
    _make_tool("send_email", "Send an email to a recipient", {"to": {"type": "string"}, "body": {"type": "string"}}),
    _make_tool("read_file", "Read contents of a file from disk", {"path": {"type": "string"}}),
    _make_tool("write_file", "Write contents to a file on disk", {"path": {"type": "string"}, "content": {"type": "string"}}),
    _make_tool("list_directory", "List files in a directory", {"path": {"type": "string"}}),
    _make_tool("run_sql", "Execute a SQL query against the database", {"query": {"type": "string"}}),
    _make_tool("create_task", "Create a new task in the project tracker", {"title": {"type": "string"}}),
    _make_tool("git_commit", "Create a git commit with a message", {"message": {"type": "string"}}),
    _make_tool("deploy_service", "Deploy a service to production", {"service": {"type": "string"}}),
    _make_tool("get_stock_price", "Get the current stock price", {"ticker": {"type": "string"}}),
    _make_tool("translate_text", "Translate text between languages", {"text": {"type": "string"}, "target_lang": {"type": "string"}}),
    _make_tool("resize_image", "Resize an image to given dimensions", {"path": {"type": "string"}, "width": {"type": "integer"}}),
    _make_tool("compress_file", "Compress a file using gzip", {"path": {"type": "string"}}),
    _make_tool("send_slack_message", "Send a message to a Slack channel", {"channel": {"type": "string"}, "text": {"type": "string"}}),
]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestToolSchemaFilter:
    def setup_method(self):
        self.filter = ToolSchemaFilter(max_tools=5)

    def test_no_filtering_when_under_limit(self):
        tools = SAMPLE_TOOLS[:3]
        messages = [{"role": "user", "content": "Hello"}]
        filtered, removed = self.filter.filter(tools, messages)
        assert len(filtered) == 3
        assert removed == 0

    def test_filters_to_max_tools(self):
        messages = [{"role": "user", "content": "What's the weather like today?"}]
        filtered, removed = self.filter.filter(SAMPLE_TOOLS, messages)
        assert len(filtered) == 5
        assert removed == 10

    def test_relevant_tools_kept(self):
        """Weather query → get_weather should be in top 5."""
        messages = [{"role": "user", "content": "Check the weather forecast for New York"}]
        filtered, removed = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "get_weather" in tool_names

    def test_file_operations_relevant(self):
        """File operation query → read_file and list_directory should appear."""
        messages = [{"role": "user", "content": "Read the config file and list the directory contents"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names

    def test_database_query_relevant(self):
        """SQL query → run_sql should appear."""
        messages = [{"role": "user", "content": "Run a SQL query to get all users from the database"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "run_sql" in tool_names

    def test_preserves_original_order(self):
        """Filtered tools should be in the same relative order as the original."""
        messages = [{"role": "user", "content": "deploy the service and commit the code"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        original_indices = []
        for t in filtered:
            for i, orig in enumerate(SAMPLE_TOOLS):
                if orig["name"] == t["name"]:
                    original_indices.append(i)
                    break
        assert original_indices == sorted(original_indices)

    def test_empty_tools(self):
        filtered, removed = self.filter.filter([], [{"role": "user", "content": "hi"}])
        assert filtered == []
        assert removed == 0

    def test_uses_most_recent_message(self):
        """Should use the last user message for relevance scoring."""
        messages = [
            {"role": "user", "content": "Tell me about the stock market"},
            {"role": "assistant", "content": "The stock market is..."},
            {"role": "user", "content": "Now check the weather in London"},
        ]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        # Last message is about weather → get_weather should be kept
        assert "get_weather" in tool_names

    def test_max_tools_configurable(self):
        f = ToolSchemaFilter(max_tools=3)
        messages = [{"role": "user", "content": "hello"}]
        filtered, removed = f.filter(SAMPLE_TOOLS, messages)
        assert len(filtered) == 3
        assert removed == 12

    def test_empty_messages_fallback(self):
        """No messages → return all tools (safe fallback)."""
        filtered, removed = self.filter.filter(SAMPLE_TOOLS, [])
        assert filtered == SAMPLE_TOOLS
        assert removed == 0

    def test_no_user_message_fallback(self):
        """Only assistant messages → return all tools (safe fallback)."""
        messages = [{"role": "assistant", "content": "How can I help?"}]
        filtered, removed = self.filter.filter(SAMPLE_TOOLS, messages)
        assert filtered == SAMPLE_TOOLS
        assert removed == 0

    def test_score_log_populated(self):
        """score_log should contain (name, score, status) for each tool."""
        messages = [{"role": "user", "content": "Check the weather forecast"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        assert len(self.filter.score_log) == len(SAMPLE_TOOLS)
        kept_count = sum(1 for _, _, status in self.filter.score_log if status == "kept")
        dropped_count = sum(1 for _, _, status in self.filter.score_log if status == "dropped")
        assert kept_count == 5
        assert dropped_count == 10

    def test_model_shorthand_minilm(self):
        """'minilm' shorthand should be accepted."""
        f = ToolSchemaFilter(model="minilm")
        assert f.model == "all-MiniLM-L6-v2"

    def test_list_content_user_message(self):
        """Should handle list-format content in the user message."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check the weather in Paris"},
                ],
            }
        ]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "get_weather" in tool_names

    def test_fallback_on_exception(self):
        """fallback_to_all=True should return all tools if embedding fails."""
        f = ToolSchemaFilter(max_tools=5, fallback_to_all=True)
        messages = [{"role": "user", "content": "hello"}]

        with patch("contextprune.embeddings.embed", side_effect=RuntimeError("model error")):
            filtered, removed = f.filter(SAMPLE_TOOLS, messages)

        # Should fall back to returning all tools
        assert len(filtered) == len(SAMPLE_TOOLS)
        assert removed == 0

    def test_score_by_name_and_description(self):
        """score_by='name_and_description' should still work."""
        f = ToolSchemaFilter(max_tools=5, score_by="name_and_description")
        messages = [{"role": "user", "content": "Check the weather forecast"}]
        filtered, removed = f.filter(SAMPLE_TOOLS, messages)
        assert len(filtered) == 5
        assert removed == 10


# ---------------------------------------------------------------------------
# Integration tests (require real embedding model)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestToolSchemaFilterIntegration:
    """Real embedding tests. Run with: pytest -m integration"""

    def setup_method(self):
        self.filter = ToolSchemaFilter(
            max_tools=5,
            model="all-MiniLM-L6-v2",
        )

    def test_weather_query_selects_weather_tool(self):
        messages = [{"role": "user", "content": "What's the weather like in Tokyo today?"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "get_weather" in tool_names, f"Expected get_weather in {tool_names}"

    def test_sql_query_selects_run_sql(self):
        messages = [{"role": "user", "content": "Execute a SQL query to find all users"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "run_sql" in tool_names

    def test_financial_data_recall(self):
        """Test case from task spec: 'fetch live stock prices' should select financial tool."""
        financial_tools = SAMPLE_TOOLS + [
            _make_tool(
                "financial_data_fetcher",
                "Fetch live stock prices and financial market data in real time",
                {"ticker": {"type": "string"}, "exchange": {"type": "string"}},
            )
        ]
        assert len(financial_tools) == 16, "Should have 16 tools for this test"

        # Use a filter that keeps top 5 from 16
        f = ToolSchemaFilter(max_tools=5, model="all-MiniLM-L6-v2")
        messages = [{"role": "user", "content": "fetch live stock prices"}]
        filtered, _ = f.filter(financial_tools, messages)
        tool_names = [t["name"] for t in filtered]
        assert "financial_data_fetcher" in tool_names, (
            f"financial_data_fetcher should be in top 5, got: {tool_names}"
        )

    def test_original_order_preserved_integration(self):
        """Original schema order preserved in filtered results."""
        messages = [{"role": "user", "content": "Search the web and read a file"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        original_indices = [
            next(i for i, t in enumerate(SAMPLE_TOOLS) if t["name"] == f["name"])
            for f in filtered
        ]
        assert original_indices == sorted(original_indices)
