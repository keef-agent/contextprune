# ContextPrune + PydanticAI

PydanticAI accepts a custom `base_url` on its `AnthropicModel`. One line to point it at the proxy.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
pip install pydantic-ai
```

## Integration

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel(
    "claude-sonnet-4-6",
    base_url="http://localhost:8899",
    # api_key is read from ANTHROPIC_API_KEY env var
)

agent = Agent(
    model,
    system_prompt="You are a concise, accurate assistant.",
)

result = agent.run_sync("What are the three laws of thermodynamics?")
print(result.data)
```

## With structured output

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    conclusion: str

model = AnthropicModel(
    "claude-sonnet-4-6",
    base_url="http://localhost:8899",
)

agent = Agent(
    model,
    result_type=Summary,
    system_prompt="Summarize the provided text into structured form.",
)

result = agent.run_sync(
    "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative phosphorylation. "
    "ATP is the primary energy currency of the cell."
)
print(result.data.key_points)
```

## Async

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel("claude-sonnet-4-6", base_url="http://localhost:8899")
agent = Agent(model)

async def main():
    result = await agent.run("Explain gradient descent in one paragraph.")
    print(result.data)

asyncio.run(main())
```

## Notes

- `base_url` supported in `pydantic-ai >= 0.0.14`
- Structured output uses tool-calling internally — ContextPrune tool filtering may reduce token usage further
- Stats logged to `~/.contextprune/stats.jsonl`

See real-world results in our benchmarks.
