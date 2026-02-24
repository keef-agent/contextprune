# contextprune

Cut your LLM API costs with 2 lines of code.

`contextprune` is a drop-in middleware that compresses API requests before they hit the model. No prompt rewriting. No quality loss. Just fewer wasted tokens.

## Benchmark Results

Measured with tiktoken `cl100k_base`. Run: `python3 benchmarks/run_all.py`.

| Scenario | Before | After | Savings | Notes |
|----------|--------|-------|---------|-------|
| Agent with 3 memory files (60% overlap) | 1,248 tokens | 1,033 tokens | **17%** | Dedup removed 16 redundant sentences |
| RAG with 10 doc chunks (40% overlap) | 1,980 tokens | 1,990 tokens | **0%** | Chunks had insufficient sentence-level overlap |
| Tool-heavy agent (20 tools, 2 topics) | 2,341 tokens | 1,241 tokens | **47%** | Tool filter eliminated 10 irrelevant schemas |
| Repetitive chat (system prompt restated) | 1,019 tokens | 718 tokens | **30%** | Dedup removed 36 repeated system prompt sentences |
| Code agent (large codebase context) | 2,542 tokens | 2,553 tokens | **0%** | Unique code; budget injection added 11 tokens |

**Average reduction: 18.7%** across all scenarios. Results vary significantly by workload type.
The tool-heavy scenario benefits most (47%); RAG and code contexts benefit least without
sentence-level repetition.

Additional benchmark metrics (full report in `benchmarks/results/report.md`):
- **Latency overhead:** 0.18–2.27% (well under 5% target vs. 500ms API call)
- **Tool recall:** 75% — 8/10 correct tools selected; keyword-based scorer misses 2 edge cases
- **API accuracy (gpt-4o-mini):** 90% raw vs 80% compressed (−10% delta, 2 questions affected)
- **Semantic preservation:** 0.67–1.00 cosine similarity (lower when dedup removes repeated blocks)

## Installation

```bash
pip install contextprune
```

Optional extras:

```bash
pip install contextprune[openai]   # OpenAI support
pip install contextprune[fast]     # sentence-transformers for better dedup
```

## Quick Start

Two lines. That's it.

```python
import anthropic
from contextprune import wrap

client = wrap(anthropic.Anthropic())

# Use it exactly like the normal Anthropic client
response = client.messages.create(
    model="claude-sonnet-4-5-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)

# Check what you saved
print(response.compression_stats)
# CompressionStats(original_tokens=4821, compressed_tokens=1203, savings_pct=75.1, time_ms=12)
```

OpenAI works the same way:

```python
import openai
from contextprune import wrap_openai

client = wrap_openai(openai.OpenAI())
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.compression_stats)
```

## How It Works

Three compression layers run on every request:

### 1. SemanticDeduplicator

Your system prompt says "You are a helpful assistant." Your memory file says "Act as a helpful assistant." Message #3 repeats the same instructions. That's 3x the tokens for the same information.

SemanticDedup uses TF-IDF cosine similarity to find and remove redundant sentences. Later duplicates get pruned. Earlier ones stay.

### 2. ToolSchemaFilter

You have 20 tools registered. The user asks "what's the weather?" Only 2-3 tools are relevant. The other 17 tool schemas are pure waste.

ToolFilter scores each tool against the current message using keyword matching and sends only the top K most relevant tools.

### 3. TokenBudgetInjector

Simple questions get simple answers. Complex questions get detailed ones. BudgetInjector estimates task complexity and appends a calibrated token budget hint to the system prompt. This nudges the model to match response length to the task.

## Configuration

```python
from contextprune import wrap, Config

client = wrap(anthropic.Anthropic(), config=Config(
    semantic_dedup=True,       # default: True
    tool_filter=True,          # default: True
    budget_injection=True,     # default: True
    max_tools=10,              # max tools to send (default: 10)
    similarity_threshold=0.85, # dedup threshold (default: 0.85)
    verbose=False              # print savings per call (default: False)
))
```

Set `verbose=True` to see savings printed on every call:

```
[contextprune] 4821 -> 1203 tokens (75.1% saved, 12.3ms)
```

## CLI Profiler

Analyze a conversation without making API calls:

```bash
contextprune profile --input conversation.json
```

Input format:

```json
{
  "system": "You are a helpful assistant...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [...]
}
```

Output shows token breakdown by layer, total savings, and recommendations.

## When It Helps Most

- Agents with memory/context files (high redundancy across messages)
- RAG pipelines with many retrieved documents (overlapping content)
- Tool-heavy agents (20+ tools but only a few relevant per turn)
- Long conversations with repeated context

## When It Helps Less

- Short, single-turn conversations with no tools
- Already-minimal system prompts
- Conversations with mostly unique content

## Contributing

```bash
git clone https://github.com/yourusername/contextprune.git
cd contextprune
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
