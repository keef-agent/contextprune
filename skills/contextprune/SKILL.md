---
name: contextprune
description: Start, stop, and manage the ContextPrune semantic deduplication proxy. Use when: starting the proxy before an agent session, checking compression stats, configuring threshold or target provider, or wiring ContextPrune into OpenClaw agent config. The proxy intercepts LLM API calls and removes semantically redundant content before forwarding to the model.
---

# ContextPrune Skill

ContextPrune is a local semantic deduplication proxy. It intercepts LLM API calls, strips redundant sentences from context using sentence embeddings, and forwards the cleaned request. Measured 46% token reduction on live agentic sessions.

## Quick Start

```bash
# Install
pip install contextprune

# Start the proxy
contextprune serve --port 8899
```

Then set the base URL env var for your provider:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8899   # Anthropic
export OPENAI_BASE_URL=http://localhost:8899       # OpenAI / Grok / OpenRouter
```

That's all. Every LLM call from any framework now goes through ContextPrune.

## Wire into OpenClaw

In `~/.openclaw/openclaw.json`, add `baseUrl` under the provider you want to compress:

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

For the `openai-codex` provider (Codex OAuth subscription):

```json
{
  "models": {
    "providers": {
      "openai": {
        "baseUrl": "http://localhost:8899"
      }
    }
  }
}
```

Restart OpenClaw after editing. OAuth Bearer tokens pass through unchanged — only message content is deduplicated.

## Check Stats

```bash
# Last 20 requests
cat ~/.contextprune/stats.jsonl | tail -20

# Quick summary (tokens saved today)
cat ~/.contextprune/stats.jsonl | python3 -c "
import sys, json
lines = [json.loads(l) for l in sys.stdin if l.strip()]
saved = sum(r['original_tokens'] - r['compressed_tokens'] for r in lines)
print(f'{len(lines)} requests | {saved:,} tokens saved')
"
```

## Dry-Run Test (no API key needed)

Test compression without making a real API call:

```bash
curl -X POST http://localhost:8899/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 10,
    "system": "You are a helpful assistant. Acme Corp was founded in 1987.",
    "messages": [{"role": "user", "content": "Acme Corp was founded in 1987. What year?"}],
    "__contextprune_dry_run": true
  }'
```

Returns compression stats without forwarding to the API.

## Threshold Tuning

Default threshold is **0.82** — conservative and safe.

```bash
# More aggressive (removes more, slightly higher risk)
contextprune serve --port 8899 --threshold 0.78

# More conservative (removes less, very safe)
contextprune serve --port 8899 --threshold 0.88
```

Rule of thumb:
- `0.75-0.80`: aggressive, best for highly repetitive agentic contexts
- `0.82`: default, works well for most cases
- `0.85-0.90`: conservative, use when context has intentional near-repetition (e.g., structured templates)

A redundancy guard prevents deduplication entirely when mean pairwise similarity is below 0.35 — so non-redundant contexts always pass through unchanged.

## Provider Routing

For non-Anthropic providers, point `--openai-target` at the right endpoint:

```bash
# Grok (xAI)
contextprune serve --port 8899 --openai-target https://api.x.ai

# OpenRouter
contextprune serve --port 8899 --openai-target https://openrouter.ai/api/v1

# Google Gemini (OpenAI-compatible mode)
contextprune serve --port 8899 --openai-target https://generativelanguage.googleapis.com/v1beta/openai
```

Full provider reference: `references/providers.md`

## References

- Science / how it works / papers: `references/science.md`
- Provider routing details: `references/providers.md`
