---
name: contextprune
description: Start, stop, and manage the ContextPrune semantic deduplication proxy for Codex sessions. Use when: starting the proxy before a Codex session, checking compression stats, or configuring threshold and provider routing. ContextPrune intercepts Codex API calls and removes semantically redundant content before forwarding to the model. No code changes required.
---

# ContextPrune Skill — Codex

ContextPrune is a local semantic deduplication proxy. It intercepts LLM API calls,
strips redundant sentences from context using sentence embeddings, and forwards the
cleaned request. Measured 46% token reduction on live agentic sessions.

## Quick Start

```bash
# Install
pip install git+https://github.com/keef-agent/contextprune.git

# Start the proxy (targeting Codex / OpenAI Responses API)
contextprune serve --port 8899 --openai-target https://api.openai.com
```

Then set the base URL so Codex routes through ContextPrune:

```bash
export OPENAI_BASE_URL=http://localhost:8899
```

Restart the Codex session after setting the env var.

## Wire into Codex Config

In your Codex workspace config (`~/.codex/config.json` or equivalent), set the
`baseUrl` for the OpenAI provider:

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

For the `openai-codex` OAuth subscription provider, the proxy intercepts the
`/v1/responses` endpoint transparently. OAuth Bearer tokens pass through unchanged. Only message content is deduplicated.

## Check Stats

```bash
# Last 20 requests
cat ~/.contextprune/stats.jsonl | tail -20

# Quick summary
cat ~/.contextprune/stats.jsonl | python3 -c "
import sys, json
lines = [json.loads(l) for l in sys.stdin if l.strip()]
saved = sum(r['original_tokens'] - r['compressed_tokens'] for r in lines)
pct = (1 - sum(r['compressed_tokens'] for r in lines) / max(sum(r['original_tokens'] for r in lines), 1)) * 100
print(f'{len(lines)} requests | {saved:,} tokens saved | {pct:.1f}% avg reduction')
"
```

## Dry-Run Test

Test compression without making a real API call:

```bash
curl -X POST http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d '{
    "model": "o3-mini",
    "messages": [
      {"role": "system", "content": "You are a coding assistant. Acme Corp was founded in 1987."},
      {"role": "user", "content": "Acme Corp was founded in 1987. What year was that?"}
    ],
    "__contextprune_dry_run": true
  }'
```

Returns compression stats without forwarding to the API.

## Threshold Tuning

Default threshold is **0.82** — conservative and safe.

```bash
# More aggressive (best for highly repetitive agentic contexts)
contextprune serve --port 8899 --openai-target https://api.openai.com --threshold 0.78

# More conservative (use when context has intentional near-repetition)
contextprune serve --port 8899 --openai-target https://api.openai.com --threshold 0.88
```

Non-redundant contexts pass through unchanged. Nothing is removed if no chunks exceed the similarity threshold.

## System Prompt Safety

ContextPrune never modifies the system prompt by default (`protect_system=True`).
All system chunks are added to the deduplication pool (so redundant content in
messages is still stripped), but the system prompt itself is returned unchanged.
This prevents compression from silently causing the model to ignore instructions.

## References

- Science / how it works / papers: see `../openclaw/references/science.md`
- Provider routing details: see `../openclaw/references/providers.md`
