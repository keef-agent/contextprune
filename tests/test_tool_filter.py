"""Tests for ToolSchemaFilter."""

from contextprune.tool_filter import ToolSchemaFilter


def _make_tool(name: str, description: str, params: dict = None) -> dict:
    """Helper to create a tool schema."""
    tool = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": params or {},
        },
    }
    return tool


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
        messages = [{"role": "user", "content": "Check the weather forecast for New York"}]
        filtered, removed = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "get_weather" in tool_names

    def test_file_operations_relevant(self):
        messages = [{"role": "user", "content": "Read the config file and list the directory contents"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names

    def test_database_query_relevant(self):
        messages = [{"role": "user", "content": "Run a SQL query to get all users from the database"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "run_sql" in tool_names

    def test_preserves_original_order(self):
        messages = [{"role": "user", "content": "deploy the service and commit the code"}]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        # Filtered tools should be in the same relative order as the original
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

    def test_uses_recent_messages(self):
        """Should prioritize recent user messages for relevance."""
        messages = [
            {"role": "user", "content": "Tell me about the stock market"},
            {"role": "assistant", "content": "The stock market is..."},
            {"role": "user", "content": "Now check the weather in London"},
        ]
        filtered, _ = self.filter.filter(SAMPLE_TOOLS, messages)
        tool_names = [t["name"] for t in filtered]
        assert "get_weather" in tool_names

    def test_max_tools_configurable(self):
        f = ToolSchemaFilter(max_tools=3)
        messages = [{"role": "user", "content": "hello"}]
        filtered, removed = f.filter(SAMPLE_TOOLS, messages)
        assert len(filtered) == 3
        assert removed == 12
