# ContextPrune + Claude Code

Claude Code reads `ANTHROPIC_BASE_URL` from the environment. Set it to the ContextPrune proxy and every Claude Code API call is compressed before hitting Anthropic.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Launch Claude Code with the proxy**

```bash
ANTHROPIC_BASE_URL=http://localhost:8899 claude
```

## Persistent configuration

Export in your shell profile so every `claude` session uses the proxy:

```bash
# ~/.bashrc or ~/.zshrc
export ANTHROPIC_BASE_URL=http://localhost:8899
```

Then just run:

```bash
claude
```

## API key mode

Works the same regardless of auth method:

```bash
# API key via env var
export ANTHROPIC_API_KEY=your-api-key
export ANTHROPIC_BASE_URL=http://localhost:8899
claude

# Or inline
ANTHROPIC_BASE_URL=http://localhost:8899 ANTHROPIC_API_KEY=your-api-key claude
```

## Verify it's working

After a session, check the stats log:

```bash
tail -n 10 ~/.contextprune/stats.jsonl
```

Example output:

```json
{"timestamp": "2026-02-26T20:50:44Z", "model": "claude-sonnet-4-6", "original_tokens": 12450, "compressed_tokens": 6743, "ratio": 0.54, "sentences_removed": 95}
```

The proxy also prints per-request stats to its console:

```
[ContextPrune] Q1 ratio=0.54 removed=95sents saved=5707tok
```

## Why it helps

Claude Code sessions accumulate large context fast: system prompts, file contents, tool outputs, and conversation history all pile up. When the same file or instruction appears in multiple messages, ContextPrune removes the duplicates before they reach the API.

Measured on a live 2-hour Claude Code session: **46% token reduction, 95 sentences removed**.

## Notes

- `ANTHROPIC_BASE_URL` is the official Anthropic SDK override for the base endpoint
- Streaming is passed through to Anthropic unchanged; deduplication runs on non-streaming requests
- Stats logged to `~/.contextprune/stats.jsonl`

See real-world results in our benchmarks.
