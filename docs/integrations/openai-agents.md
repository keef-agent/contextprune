# ContextPrune + OpenAI Agents SDK

The OpenAI Agents SDK supports custom base URLs via `AsyncOpenAI`. Point it at the ContextPrune proxy before any agent runs.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
pip install openai-agents openai
```

## Integration

```python
from agents import set_default_client, Agent, Runner
from openai import AsyncOpenAI

# Route all agent calls through ContextPrune
set_default_client(
    AsyncOpenAI(
        base_url="http://localhost:8899",
        # api_key is read from ANTHROPIC_API_KEY or OPENAI_API_KEY env var
    )
)

agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="claude-sonnet-4-6",
)

import asyncio

async def main():
    result = await Runner.run(agent, "Summarize the theory of relativity in 3 sentences.")
    print(result.final_output)

asyncio.run(main())
```

## Per-agent client (override default)

```python
from agents import Agent, Runner
from openai import AsyncOpenAI

pruned_client = AsyncOpenAI(base_url="http://localhost:8899")

agent = Agent(
    name="research-agent",
    instructions="You are a research assistant with deep memory context.",
    model="claude-sonnet-4-6",
    client=pruned_client,
)
```

## Notes

- `set_default_client` must be called before any agent is instantiated
- ContextPrune deduplication is most effective when agent instructions + memory are verbose and repetitive
- Streaming is passed through unchanged; compression runs on non-streaming requests
- Stats logged to `~/.contextprune/stats.jsonl`

See real-world results in our benchmarks.
