# ContextPrune

**Cut your LLM token usage by 40-60% with a single environment variable.**

ContextPrune is a local semantic deduplication proxy. It sits between your agent and the LLM API, strips repeated information from context before it hits the model, and forwards the cleaned request. No code changes. No data leaves your machine except to the LLM you already trust.

**Real results measured on live sessions:**
- 2-hour AI agent session (OpenClaw): **46% token reduction**
- Agentic context (system prompt + memory + tool outputs): **36-45% reduction**
- Non-redundant contexts: **0% reduction** *(correctly passes through unchanged)*

**Why it matters:**
- API users: direct reduction in your monthly bill
- Claude Code / Codex subscription users: fewer tokens used per turn = significantly more turns before hitting daily/hourly usage limits
- Long-running agents: context compaction and session resets happen much less often

---

## How It Works

### The Problem

Real agent contexts repeat the same information 3-5x. A typical long-running agent sends:
- System prompt (role, capabilities, rules)
- Memory files (recent context, past decisions)
- Tool outputs (results from previous steps)
- Conversation history

All of these overlap heavily. The model sees the same facts stated four different ways and charges you for every token.

### The Solution: Semantic Deduplication

ContextPrune encodes every sentence in your context using a lightweight sentence embedding model ([paraphrase-MiniLM-L6-v2](https://www.sbert.net/), 22M params, runs locally in <50ms). It then computes pairwise cosine similarity between sentences across all messages and removes near-duplicates above a configurable threshold (default: 0.82).

This approach is grounded in established NLP research:

- **Semantic deduplication via embeddings**: Broadly validated in retrieval-augmented generation literature (RAG dedup). Similar similarity thresholds (0.8-0.85) are standard in RAG pipelines for chunk deduplication.
- **LLMLingua** ([Jiang et al., 2023](https://arxiv.org/abs/2310.05736)): Demonstrated 20x compression with <1.5% accuracy loss using token-level importance scoring. ContextPrune uses sentence-level semantic similarity instead — safer for agent contexts where removing wrong tokens causes hallucination.
- **ACON** ([Zhong et al., 2024](https://arxiv.org/abs/2406.06548)): Showed 26-54% context reduction specifically in agentic workloads using semantic redundancy detection — the most directly relevant prior work to what ContextPrune does.
- **Token Elasticity / TALE** ([Ivgi et al., 2024](https://arxiv.org/abs/2407.07955)): Quantified that models require explicit, calibrated token budgets — vague instructions like "be concise" don't work. ContextPrune removes redundancy at the infrastructure level instead, bypassing this problem entirely.

### What Gets Removed

Only sentences that are semantically near-identical to something already in the context. Unique information, reasoning chains, and code are never touched — they have no semantic near-duplicates. ContextPrune has a conservative default threshold and a redundancy guard (mean pairwise similarity must exceed 0.35 to even attempt deduplication) so it passes through unchanged when context isn't redundant.

---

## Quick Start

### Install

```bash
pip install contextprune
```

### Start the proxy

```bash
contextprune serve --port 8899
```

### Set one environment variable

```bash
export ANTHROPIC_BASE_URL=http://localhost:8899   # Anthropic API
export OPENAI_BASE_URL=http://localhost:8899       # OpenAI API
```

That's it. Every LLM API call from any framework now goes through ContextPrune first.

Works with: LangChain, LangGraph, CrewAI, AG2/AutoGen, OpenAI Agents SDK, PydanticAI, Google ADK, Mastra, Vercel AI SDK, LlamaIndex, Claude Code, and any other framework that respects these env vars.

---

## Setup by Use Case

### LLM API users (OpenAI, Anthropic, Grok, Google, OpenRouter)

ContextPrune supports all three major API formats:
- `POST /v1/messages` — Anthropic Messages API
- `POST /v1/chat/completions` — OpenAI Chat Completions (also Grok, OpenRouter, Google Gemini compat)
- `POST /v1/responses` — OpenAI Responses API (used by OpenAI Agents SDK and Codex)

```bash
contextprune serve --port 8899

# Anthropic
export ANTHROPIC_BASE_URL=http://localhost:8899

# OpenAI / Grok / OpenRouter
export OPENAI_BASE_URL=http://localhost:8899

# OpenRouter (point to OpenRouter as target)
contextprune serve --port 8899 --openai-target https://openrouter.ai/api/v1

# Grok (xAI)
contextprune serve --port 8899 --openai-target https://api.x.ai

# Google Gemini (OpenAI-compatible mode)
contextprune serve --port 8899 --openai-target https://generativelanguage.googleapis.com/v1beta/openai
```

### Claude Code

Works with both API key and claude.ai OAuth subscription:

```bash
contextprune serve --port 8899
export ANTHROPIC_BASE_URL=http://localhost:8899
claude
```

Claude Code's OAuth Bearer token passes through to api.anthropic.com unchanged. ContextPrune only touches message content.

**Subscription benefit:** Claude Code compacts context aggressively when usage builds up. With 40%+ fewer tokens per request, compaction events happen significantly less often — you keep more conversation history alive, longer.

### OpenClaw

In `~/.openclaw/openclaw.json`, under the provider you want to compress:

```json
"models": {
  "providers": {
    "anthropic": {
      "baseUrl": "http://localhost:8899"
    }
  }
}
```

Restart OpenClaw. Measured 46% reduction on a live 2-hour session.

### OpenClaw with OAuth subscription (Codex / openai-codex provider)

The `openai-codex` provider uses OAuth Bearer tokens but still calls `api.openai.com/v1/responses` — the standard Responses API. ContextPrune intercepts it:

```json
"models": {
  "providers": {
    "openai": {
      "baseUrl": "http://localhost:8899"
    }
  }
}
```

Your OAuth token passes through unchanged. Only the message content is deduplicated.

**Subscription benefit:** Codex subscription plans have hourly/daily usage limits measured in tokens. With 40%+ compression, you get significantly more agent turns before hitting those limits.

### Agent Skills (OpenClaw, Codex, and agentskills.io-compatible frameworks)

ContextPrune ships a skill that works across any framework built on the [open agent skills standard](https://agentskills.io) — the same format used by OpenClaw and Codex.

**Install the skill:**
```bash
# For Codex
mkdir -p ~/.agents/skills
cp -r skills/contextprune ~/.agents/skills/

# For OpenClaw
mkdir -p ~/.openclaw/workspace/skills
cp -r skills/contextprune ~/.openclaw/workspace/skills/
```

Once installed, you can invoke it explicitly (`$contextprune` in Codex, or it triggers implicitly when relevant) or let the agent pick it up automatically. The skill instructs the agent to check proxy status, start it if needed, and report compression stats.

The skill directory is at `skills/contextprune/` in this repo. One file, works everywhere.

---

## Supported Providers

| Provider | Format | Works |
|----------|--------|-------|
| Anthropic API | `/v1/messages` | ✅ |
| Claude Code (API key or OAuth) | `/v1/messages` | ✅ |
| OpenAI API | `/v1/responses` + `/v1/chat/completions` | ✅ |
| Grok (xAI) | OpenAI-compatible | ✅ |
| OpenRouter | OpenAI-compatible | ✅ |
| Google Gemini | OpenAI-compatible mode | ✅ |
| OpenClaw (any provider) | configurable baseUrl | ✅ |
| OpenAI Codex subscription OAuth | `/v1/responses` | ✅ |
| claude.ai web / ChatGPT.com | proprietary internal endpoints | ❌ |

---

## Configuration

```bash
# Threshold: lower = more aggressive, higher = more conservative (default: 0.82)
contextprune serve --port 8899 --threshold 0.80

# Disable stats logging
contextprune serve --port 8899 --no-log

# Enable verbose per-request output
contextprune serve --port 8899 --log
```

### Dry-run test (no API key needed)

Add `"__contextprune_dry_run": true` to any request body to skip forwarding and get compression stats back directly:

```bash
curl -X POST http://localhost:8899/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 10,
    "system": "You are a helpful assistant. Acme Corp was founded in 1987 and makes enterprise software.",
    "messages": [{"role": "user", "content": "Acme Corp was founded in 1987. They make enterprise software. What year was Acme Corp founded?"}],
    "__contextprune_dry_run": true
  }'
```

Response:
```json
{"contextprune": {"original_tokens": 49, "compressed_tokens": 29, "ratio": 0.59, "sentences_removed": 2}}
```

### Stats

All requests are logged to `~/.contextprune/stats.jsonl`:

```bash
cat ~/.contextprune/stats.jsonl | tail -20
```

```json
{"timestamp": "2026-02-26T20:50:44Z", "model": "claude-sonnet-4-6", "original_tokens": 49, "compressed_tokens": 29, "ratio": 0.59, "sentences_removed": 2}
```

### Architecture

```
Agent Framework → localhost:8899 → ContextPrune dedup → api.anthropic.com
                                          ↓
                                   logs to ~/.contextprune/stats.jsonl
```

Streaming requests (`"stream": true`) are passed through unchanged. Deduplication only runs on non-streaming requests.

---

## License

MIT. Local only — no data sent anywhere except the LLM you configure.
