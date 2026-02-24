"""Integration tests for the full compression pipeline.

Uses mock Anthropic client -- no real API calls.
Embedding model is also mocked to avoid downloading 137MB in CI.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from contextprune import Config, wrap
from contextprune.core import wrap_openai
from contextprune.stats import CompressionStats
from contextprune.tokenizer import count_message_tokens, count_system_tokens, count_tools_tokens

# ---------------------------------------------------------------------------
# Embedding mock — same keyword-group approach as test_dedup.py / test_tool_filter.py
# ---------------------------------------------------------------------------

_KEYWORD_GROUPS = [
    (["postgresql", "database", "port 5432", "fastapi", "framework"], 0),
    (["weather", "forecast"], 1),
    (["python", "code", "pytest", "tests", "testing", "type hints", "type annotations", "type hint"], 2),
    (["file", "read", "write", "directory"], 3),
    (["sql", "query"], 4),
    (["web", "search"], 5),
    (["email", "calendar"], 6),
    (["helpful", "assistant", "expert", "software"], 7),
    (["docker", "deployment", "deploy"], 8),
    (["react", "frontend", "backend", "authentication", "login", "endpoint", "api", "fastapi", "credential"], 9),
    (["stock", "price", "financial", "market"], 10),
    (["git", "commit", "diff"], 11),
    (["run", "command", "shell", "bash"], 12),
]

_DIM = 20
_embed_cache: dict = {}
_unique_counter = [len(_KEYWORD_GROUPS)]


def _mock_embed(texts, model_name, prefix=""):
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
def patch_embeddings(monkeypatch):
    """Mock embedding functions for the full pipeline integration tests."""
    monkeypatch.setattr("contextprune.embeddings.embed", _mock_embed)
    monkeypatch.setattr("contextprune.embeddings.cosine_similarity", _mock_cosine_sim)
    _embed_cache.clear()
    _unique_counter[0] = len(_KEYWORD_GROUPS)


def _make_tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": f"Input for {name}"},
            },
        },
    }


# Realistic agent context
SYSTEM_PROMPT = (
    "You are an expert software development assistant. "
    "You help users write, debug, and review code. "
    "You have access to the user's codebase and can read and write files. "
    "Always follow best practices for code quality and security. "
    "When writing code, follow the project's existing conventions. "
    "You are an expert in Python, JavaScript, TypeScript, and Go."
)

MEMORY_CONTENT = (
    "Project context: The user is working on a Python web application. "
    "The application uses FastAPI for the backend and React for the frontend. "
    "The database is PostgreSQL running on port 5432. "
    "You are an expert software development assistant. "  # Redundant with system
    "You help users write, debug, and review code. "  # Redundant with system
    "Always follow best practices for code quality and security. "  # Redundant with system
    "The user prefers type hints in Python code. "
    "Tests should use pytest. "
    "The project uses Docker for deployment."
)

MESSAGES = [
    {"role": "user", "content": MEMORY_CONTENT},
    {
        "role": "assistant",
        "content": "I understand the project context. How can I help you today?",
    },
    {
        "role": "user",
        "content": (
            "The application uses FastAPI for the backend. "  # Redundant
            "I need help writing a new API endpoint for user authentication. "
            "The database is PostgreSQL running on port 5432. "  # Redundant
            "The endpoint should handle login with email and password."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "I'll help you create an authentication endpoint. "
            "Since you're using FastAPI and PostgreSQL, I'll write a login endpoint "
            "that validates credentials against the database."
        ),
    },
    {
        "role": "user",
        "content": (
            "Great. Also remember that we use pytest for testing. "  # Redundant
            "Please also write tests for the authentication endpoint. "
            "Make sure to follow best practices for code quality and security. "  # Redundant
            "The tests should cover both successful and failed login attempts."
        ),
    },
    {
        "role": "assistant",
        "content": "I'll write both the endpoint and comprehensive tests.",
    },
    {
        "role": "user",
        "content": (
            "One more thing: the user prefers type hints in Python code. "  # Redundant
            "Add proper type annotations to all functions. "
            "Also make sure the endpoint handles rate limiting."
        ),
    },
]

TOOLS = [
    _make_tool("read_file", "Read the contents of a file from the filesystem"),
    _make_tool("write_file", "Write content to a file on the filesystem"),
    _make_tool("list_directory", "List files and directories in a given path"),
    _make_tool("search_code", "Search for code patterns across the codebase"),
    _make_tool("run_tests", "Run the test suite using pytest"),
    _make_tool("run_command", "Execute a shell command"),
    _make_tool("git_diff", "Show git diff of current changes"),
    _make_tool("git_commit", "Create a git commit"),
    _make_tool("get_weather", "Get the current weather forecast"),
    _make_tool("send_email", "Send an email message"),
    _make_tool("create_calendar_event", "Create a calendar event"),
    _make_tool("search_web", "Search the web for information"),
    _make_tool("get_stock_price", "Get the current stock price for a ticker"),
    _make_tool("translate_text", "Translate text between languages"),
    _make_tool("resize_image", "Resize an image to specified dimensions"),
]


class TestFullPipeline:
    def _make_mock_client(self) -> Any:
        client = MagicMock()
        response = MagicMock()
        client.messages.create.return_value = response
        return client

    def test_compression_above_30_percent(self):
        """Full pipeline should achieve > 30% compression on redundant context."""
        client = self._make_mock_client()
        wrapped = wrap(client, config=Config(max_tools=5))

        original_tokens = (
            count_message_tokens(MESSAGES)
            + count_system_tokens(SYSTEM_PROMPT)
            + count_tools_tokens(TOOLS)
        )

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=MESSAGES,
            tools=TOOLS,
        )

        stats = response.compression_stats
        assert isinstance(stats, CompressionStats)
        assert stats.original_tokens > 0
        assert stats.compressed_tokens > 0
        assert stats.compressed_tokens < stats.original_tokens
        assert stats.savings_pct > 30, f"Expected > 30% savings, got {stats.savings_pct}%"
        assert stats.time_ms >= 0

    def test_dedup_removes_redundant_sentences(self):
        """Dedup should catch the repeated sentences in our test context."""
        client = self._make_mock_client()
        wrapped = wrap(client, config=Config(tool_filter=False, budget_injection=False))

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=MESSAGES,
        )

        stats = response.compression_stats
        assert stats.dedup_removed > 0, "Should have removed redundant sentences"

    def test_tool_filter_reduces_tools(self):
        """Tool filter should remove irrelevant tools."""
        client = self._make_mock_client()
        wrapped = wrap(client, config=Config(
            semantic_dedup=False,
            budget_injection=False,
            max_tools=5,
        ))

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1024,
            messages=MESSAGES,
            tools=TOOLS,
        )

        stats = response.compression_stats
        assert stats.tools_removed > 0

        # Verify the actual call had fewer tools
        call_kwargs = client.messages.create.call_args[1]
        assert len(call_kwargs["tools"]) == 5

    def test_budget_injection(self):
        """Budget injector should add token budget to system prompt."""
        client = self._make_mock_client()
        wrapped = wrap(client, config=Config(
            semantic_dedup=False,
            tool_filter=False,
            budget_injection=True,
        ))

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1024,
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        stats = response.compression_stats
        assert stats.budget_injected is True

        call_kwargs = client.messages.create.call_args[1]
        assert "[Token Budget:" in call_kwargs["system"]

    def test_messages_remain_valid(self):
        """After compression, messages should still have valid structure."""
        client = self._make_mock_client()
        wrapped = wrap(client)

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=MESSAGES,
            tools=TOOLS,
        )

        call_kwargs = client.messages.create.call_args[1]
        messages = call_kwargs["messages"]

        for msg in messages:
            assert "role" in msg, "Message missing 'role'"
            assert "content" in msg, "Message missing 'content'"
            assert msg["role"] in ("user", "assistant"), f"Invalid role: {msg['role']}"
            assert msg["content"], "Message has empty content"

    def test_passthrough_kwargs(self):
        """Non-compression kwargs should pass through unchanged."""
        client = self._make_mock_client()
        wrapped = wrap(client)

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=2048,
            temperature=0.7,
            messages=[{"role": "user", "content": "Hello"}],
        )

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-5-20250514"
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.7

    def test_no_tools_no_crash(self):
        """Pipeline should work fine without tools."""
        client = self._make_mock_client()
        wrapped = wrap(client)

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        stats = response.compression_stats
        assert isinstance(stats, CompressionStats)
        assert stats.tools_removed == 0

    def test_all_layers_disabled(self):
        """With all layers off, tokens should be roughly the same."""
        client = self._make_mock_client()
        wrapped = wrap(client, config=Config(
            semantic_dedup=False,
            tool_filter=False,
            budget_injection=False,
        ))

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        stats = response.compression_stats
        assert stats.original_tokens == stats.compressed_tokens
        assert stats.savings_pct == 0.0

    def test_verbose_prints(self, capsys):
        """Verbose mode should print stats."""
        client = self._make_mock_client()
        wrapped = wrap(client, config=Config(verbose=True))

        response = wrapped.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        captured = capsys.readouterr()
        assert "[contextprune]" in captured.out


class TestOpenAIWrapper:
    def _make_mock_openai_client(self) -> Any:
        client = MagicMock()
        response = MagicMock()
        client.chat.completions.create.return_value = response
        return client

    def test_openai_basic(self):
        """OpenAI wrapper should work with basic messages."""
        client = self._make_mock_openai_client()
        wrapped = wrap_openai(client)

        response = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )

        stats = response.compression_stats
        assert isinstance(stats, CompressionStats)
        assert stats.original_tokens > 0

    def test_openai_with_tools(self):
        """OpenAI wrapper should handle tool filtering."""
        client = self._make_mock_openai_client()
        wrapped = wrap_openai(client, config=Config(max_tools=3))

        openai_tools = [
            {"type": "function", "function": _make_tool(f"tool_{i}", f"Description for tool {i}")}
            for i in range(15)
        ]

        response = wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            tools=openai_tools,
        )

        stats = response.compression_stats
        assert stats.tools_removed > 0

        call_kwargs = client.chat.completions.create.call_args[1]
        assert len(call_kwargs["tools"]) == 3
        # Verify they're re-wrapped in OpenAI format
        for t in call_kwargs["tools"]:
            assert t["type"] == "function"
            assert "function" in t

    def test_client_passthrough(self):
        """Non-chat attributes should pass through to original client."""
        client = self._make_mock_openai_client()
        client.api_key = "test-key"
        wrapped = wrap_openai(client)
        assert wrapped.api_key == "test-key"
