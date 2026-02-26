# ContextPrune + NanoClaw

NanoClaw is built on the Anthropic Agents SDK. It respects the `ANTHROPIC_BASE_URL` environment variable — set it before starting NanoClaw and every agent call routes through ContextPrune.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Set the environment variable**

```bash
export ANTHROPIC_BASE_URL=http://localhost:8899
```

**3. Start NanoClaw**

```bash
nanoclaw start
# or however you launch NanoClaw in your setup
```

All Anthropic API calls now route through ContextPrune automatically.

## Inline (single session)

```bash
ANTHROPIC_BASE_URL=http://localhost:8899 nanoclaw start
```

## Persistent configuration

Add to your shell profile:

```bash
# ~/.bashrc or ~/.zshrc
export ANTHROPIC_BASE_URL=http://localhost:8899
```

Or configure it in your NanoClaw `.env` file if it exists:

```bash
# .env
ANTHROPIC_BASE_URL=http://localhost:8899
ANTHROPIC_API_KEY=your-api-key
```

## Verify

After starting a session, check the ContextPrune stats log:

```bash
tail -f ~/.contextprune/stats.jsonl
```

Each line confirms a compressed request was processed. The `ratio` field shows the compression factor.

## Why it works

NanoClaw uses the Anthropic Python SDK internally. The SDK reads `ANTHROPIC_BASE_URL` at initialization time and uses it as the API endpoint — no code changes needed.

## Notes

- Deduplication is most effective in long NanoClaw sessions where system prompts, memory files, and tool outputs repeat across turns
- Stats logged to `~/.contextprune/stats.jsonl`
- Streaming is passed through unchanged

See real-world results in our benchmarks.
