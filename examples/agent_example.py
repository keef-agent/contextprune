"""Agent example showing realistic savings with contextprune.

This demonstrates the kind of token savings you get with:
- A system prompt with repeated instructions
- Memory/context files with overlapping content
- Multiple tools (only a few relevant per turn)
"""

import anthropic
from contextprune import wrap, Config

# Realistic agent setup
SYSTEM_PROMPT = """You are an expert software development assistant.
You help users write, debug, and review code.
You have access to the user's codebase and can read and write files.
Always follow best practices for code quality and security.
When writing code, follow the project's existing conventions.
You are proficient in Python, JavaScript, TypeScript, and Go."""

# Simulated memory file (overlaps heavily with system prompt)
MEMORY = """Project context:
- The user is building a Python web API with FastAPI
- Database: PostgreSQL on port 5432
- You are an expert software development assistant
- You help users write, debug, and review code
- Always follow best practices for code quality and security
- The user prefers type hints in all Python code
- Tests use pytest with pytest-cov for coverage
- Deployment uses Docker and GitHub Actions"""

TOOLS = [
    {"name": "read_file", "description": "Read file contents", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "write_file", "description": "Write content to file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "list_dir", "description": "List directory contents", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "search_code", "description": "Search codebase for patterns", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "run_tests", "description": "Run pytest test suite", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "run_command", "description": "Execute shell command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}},
    {"name": "git_diff", "description": "Show git diff", "input_schema": {"type": "object", "properties": {}}},
    {"name": "git_commit", "description": "Create git commit", "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}}},
    {"name": "get_weather", "description": "Get weather forecast", "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}},
    {"name": "send_email", "description": "Send email", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}}},
    {"name": "calendar_event", "description": "Create calendar event", "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}}},
    {"name": "web_search", "description": "Search the web", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "stock_price", "description": "Get stock price", "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}}},
    {"name": "translate", "description": "Translate text", "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "resize_image", "description": "Resize an image", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
]

MESSAGES = [
    {"role": "user", "content": MEMORY},
    {"role": "assistant", "content": "I understand the project context. How can I help?"},
    {"role": "user", "content": "The database is PostgreSQL on port 5432. I need a new endpoint for user auth. Follow best practices for security."},
    {"role": "assistant", "content": "I'll create an authentication endpoint using FastAPI with proper security."},
    {"role": "user", "content": "Write tests too. We use pytest. Add type hints to everything."},
]

client = wrap(
    anthropic.Anthropic(),
    config=Config(
        max_tools=5,
        verbose=True,
    ),
)

# This call will compress the context before sending
response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=2048,
    system=SYSTEM_PROMPT,
    messages=MESSAGES,
    tools=TOOLS,
)

stats = response.compression_stats
print(f"\nOriginal:   {stats.original_tokens} tokens")
print(f"Compressed: {stats.compressed_tokens} tokens")
print(f"Saved:      {stats.savings_pct}%")
print(f"Time:       {stats.time_ms}ms")
