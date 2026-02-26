# ContextPrune + OpenClaw

ContextPrune cuts token usage by intercepting Anthropic API calls before they leave your machine. Measured reduction on a live 2-hour keef-direct session: **46% fewer tokens, 95 sentences removed**.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Configure OpenClaw**

In `~/.openclaw/openclaw.json`, under `models.providers.anthropic`:

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

**3. Restart OpenClaw**

Done. Every Anthropic request now runs through ContextPrune.

## What gets compressed

- Repeated sentences across system prompt, memory files, and message history
- Tool schemas irrelevant to the current turn
- Token budget hints calibrated to task complexity

## Auto-start (optional)

Add to your shell profile or a startup script:

```bash
# ~/.bashrc or systemd service
contextprune serve --port 8899 &
```

## Verify it's working

```bash
# Check the stats log after a session
tail -n 5 ~/.contextprune/stats.jsonl
```

Each line shows original tokens, compressed tokens, ratio, and sentences removed.

## Real results

- **2-hour OpenClaw session:** 46% token reduction, 95 sentences removed
- **Agentic context (system + memory + tools):** 36.6% reduction
- **Unique content with no redundancy:** 0% reduction (passes through unchanged)

See real-world results in our benchmarks.
