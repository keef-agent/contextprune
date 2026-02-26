# ContextPrune: Provider Routing Reference

## API Endpoints

ContextPrune proxies three API formats:

| Endpoint | Format | Used by |
|---|---|---|
| `POST /v1/messages` | Anthropic Messages API | Anthropic API, Claude Code, OpenClaw (anthropic provider) |
| `POST /v1/chat/completions` | OpenAI Chat Completions | OpenAI, Grok, OpenRouter, Google Gemini (compat mode) |
| `POST /v1/responses` | OpenAI Responses API | OpenAI Agents SDK, Codex CLI, OpenClaw (openai-codex provider) |

## Environment Variables

Set the appropriate env var for your provider/framework:

| Provider | Env var | Value |
|---|---|---|
| Anthropic API | `ANTHROPIC_BASE_URL` | `http://localhost:8899` |
| Claude Code | `ANTHROPIC_BASE_URL` | `http://localhost:8899` |
| OpenAI API | `OPENAI_BASE_URL` | `http://localhost:8899` |
| Grok (xAI) | `OPENAI_BASE_URL` | `http://localhost:8899` |
| OpenRouter | `OPENAI_BASE_URL` | `http://localhost:8899` |
| Google Gemini | `OPENAI_BASE_URL` | `http://localhost:8899` |

## `--openai-target` for Non-OpenAI Providers

When using the OpenAI-compatible endpoint for providers other than OpenAI, tell ContextPrune where to forward the request:

```bash
# Default (OpenAI)
contextprune serve --port 8899
# forwards to: https://api.openai.com

# Grok (xAI)
contextprune serve --port 8899 --openai-target https://api.x.ai

# OpenRouter
contextprune serve --port 8899 --openai-target https://openrouter.ai/api/v1

# Google Gemini (OpenAI-compatible mode)
contextprune serve --port 8899 --openai-target https://generativelanguage.googleapis.com/v1beta/openai
```

Anthropic target is always `https://api.anthropic.com` — not configurable via this flag.

## OpenClaw Config

### Anthropic provider

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

### OpenAI / Codex provider (OAuth subscription)

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

OAuth Bearer tokens pass through unchanged. ContextPrune strips only message content — auth headers are never modified.

## What Passes Through Unchanged

- Auth headers (`x-api-key`, `Authorization: Bearer ...`)
- Model selection (`model` field)
- Streaming requests (`"stream": true`) — passed through without deduplication
- Request metadata (temperature, max_tokens, stop sequences, etc.)
- Any context below the redundancy guard threshold (mean pairwise similarity < 0.35)
