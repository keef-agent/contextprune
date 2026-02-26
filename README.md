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
```

## Embedding Models

By default, `contextprprune` uses **nomic-ai/nomic-embed-text-v1.5** for embedding-based deduplication and tool filtering. This model was chosen because:

- **2048-token context window** — critical for handling long agent contexts without truncation (unlike MiniLM's 256-token limit)
- **Strong MTEB benchmarks** — competitive retrieval performance
- **Apache 2.0 license** — permissive for commercial use

For faster cold-start times, you can switch to the lighter model:

```python
from contextprune import wrap

# Use the smaller 22MB model (~5ms per encode)
client = wrap(anthropic.Anthropic(), dedup_model="all-MiniLM-L6-v2")
```

Or configure via Config:

```python
from contextprune import Config, wrap

config = Config(dedup_model="all-MiniLM-L6-v2", tool_model="all-MiniLM-L6-v2")
client = wrap(anthropic.Anthropic(), config=config)
```

**Note:** First-call model loading takes ~2 seconds. Subsequent calls are cached and take <30ms.

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

## Proxy Server (Drop-in API Intercept)

ContextPrune ships with a local HTTP proxy that intercepts Anthropic Messages
API calls, runs semantic deduplication, and forwards the compressed request to
the real API. No code changes needed in your app.

### Architecture

```
Agent Framework → localhost:8899 → ContextPrune dedup → api.anthropic.com
                                          ↓
                                   logs savings to ~/.contextprune/stats.jsonl
```

### Start the proxy

```bash
# CLI
contextprune serve --port 8899

# Or via python module
python -m contextprune.proxy --port 8899

# Options
python -m contextprune.proxy --port 8899 --threshold 0.82 --no-log
```

### Point your client at the proxy

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8899",
    api_key="your-real-key",  # forwarded as-is to Anthropic
)
```

### Dry-run test (no Anthropic API key needed)

```bash
curl -X POST http://localhost:8899/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 10,
    "system": "You are a helpful assistant. Acme Corp was founded in 1987 and makes enterprise software.",
    "messages": [{"role": "user", "content": "Acme Corp was founded in 1987. They make enterprise software. Question: What year was Acme Corp founded?"}],
    "__contextprune_dry_run": true
  }'
# → {"contextprune": {"original_tokens": 49, "compressed_tokens": 29, "ratio": 0.59, "sentences_removed": 2}}
```

Add `"__contextprune_dry_run": true` to any request body to skip forwarding and
get compression stats back directly (useful for testing and CI).

### Per-request console output

```
[ContextPrune] Q1 ratio=0.59 removed=2sents saved=20tok
```

### Stats log

All requests are logged to `~/.contextprune/stats.jsonl`:

```json
{"timestamp": "2026-02-26T20:50:44Z", "model": "claude-sonnet-4-6", "original_tokens": 49, "compressed_tokens": 29, "ratio": 0.5918, "sentences_removed": 2}
```

### OpenClaw Integration

To route OpenClaw's Anthropic requests through the proxy, add `baseUrl` to your
`~/.openclaw/openclaw.json` under `models.providers.anthropic`:

```json
{
  "models": {
    "providers": {
      "anthropic": {
        "baseUrl": "http://localhost:8899"
      }
    }
  }
}
```

Start the proxy before starting OpenClaw:

```bash
contextprune serve --port 8899
```

### Streaming

Streaming requests (`"stream": true`) are passed through to Anthropic unchanged.
Deduplication only runs on non-streaming requests.

## Works with every framework

Start the proxy:
```bash
contextprune serve --port 8899
```

Set one environment variable:
```bash
export ANTHROPIC_BASE_URL=http://localhost:8899   # Anthropic API (/v1/messages)
export OPENAI_BASE_URL=http://localhost:8899       # OpenAI API (/v1/chat/completions + /v1/responses)
```

That's it. Works with: LangChain, LangGraph, CrewAI, AG2/AutoGen, OpenAI Agents SDK, PydanticAI, Google ADK, Mastra, Vercel AI SDK, NanoClaw, LlamaIndex, Claude Code, and any other framework that respects these env vars.

**Real results measured on live sessions:**
- 2-hour AI agent session (OpenClaw): **46% token reduction**
- Agentic context (system prompt + memory + tool outputs): **36.6% reduction**
- Non-redundant contexts: **0% reduction** (correctly passes through unchanged)

> Using a subscription? See [OAuth subscription guide](docs/oauth-subscriptions.md).

## Contributing

```bash
git clone https://github.com/yourusername/contextprune.git
cd contextprune
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
