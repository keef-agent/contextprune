# Using ContextPrune with Subscription OAuth Tokens

ContextPrune works with OAuth subscription tokens in some scenarios.

## What works

### Claude Code (claude.ai subscription)
If you use Claude Code with your claude.ai Pro/Max subscription (OAuth mode via `claude auth login`), ContextPrune intercepts at the API level — it works because Claude Code still calls `api.anthropic.com` in standard Messages API format.

```bash
contextprune serve --port 8899
export ANTHROPIC_BASE_URL=http://localhost:8899
claude  # Claude Code with claude.ai OAuth — ContextPrune active
```

Your OAuth Bearer token passes through to Anthropic unchanged. ContextPrune only touches the message content, not auth.

**Why this matters:** Claude.ai subscription has usage limits (messages per day/hour). With 40%+ fewer tokens per request, you'll hit those limits significantly less often.

### OpenAI OAuth (Codex subscription)

If you use the OpenAI Responses API with an OAuth Bearer token (e.g. OpenClaw's openai-codex provider, or Codex CLI in API key mode):

```bash
contextprune serve --port 8899
export OPENAI_BASE_URL=http://localhost:8899
```

Your Bearer token passes through to api.openai.com unchanged. ContextPrune deduplicates the `input` array and `instructions` field before forwarding.

**Why this matters:** The Responses API is used by the OpenAI Agents SDK, Codex CLI, and any framework routing through `openai-codex`. With long agentic sessions, context repetition compounds quickly — ContextPrune removes it before it reaches the model.

### OpenRouter (any provider)
OpenRouter supports OAuth-style API keys. Set target to OpenRouter:
```bash
contextprune serve --port 8899 --openai-target https://openrouter.ai/api
export OPENAI_BASE_URL=http://localhost:8899
export OPENAI_API_KEY=your-openrouter-key  # passes through unchanged
```

## What doesn't work (yet)

| Surface | Why | Status |
|---------|-----|--------|
| claude.ai web | Internal API endpoints, not `api.anthropic.com` | Browser extension planned |
| ChatGPT.com | Internal API endpoints | Browser extension planned |
| Cursor / Windsurf | Custom OAuth endpoints | Not planned |
| GitHub Copilot | Routes through GitHub proxy | Not planned |

## Browser extension (coming)

For claude.ai web and ChatGPT.com, interception requires a browser extension that hooks into fetch/XHR calls. This is on the roadmap. Star the repo to get notified.
