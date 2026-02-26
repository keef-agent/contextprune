# ContextPrune + Google ADK

Google's Agent Development Kit respects the `ANTHROPIC_BASE_URL` environment variable when using Anthropic models. Set it before starting your ADK agent.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
pip install google-adk anthropic
```

## Integration via environment variable

```bash
export ANTHROPIC_BASE_URL=http://localhost:8899
export ANTHROPIC_API_KEY=your-anthropic-api-key

# Start your ADK agent
adk run my_agent/
```

Or inline for a single session:

```bash
ANTHROPIC_BASE_URL=http://localhost:8899 adk run my_agent/
```

## Integration in code

If you construct the Anthropic client directly in your ADK agent, pass `base_url` explicitly:

```python
import anthropic
from google.adk.agents import LlmAgent
from google.adk.models.anthropic_llm import Claude

# Option 1: env var (set before import)
import os
os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:8899"

# Option 2: explicit client
client = anthropic.Anthropic(
    base_url="http://localhost:8899",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

model = Claude(model_id="claude-sonnet-4-6", client=client)

agent = LlmAgent(
    name="my_agent",
    model=model,
    instruction="You are a helpful assistant.",
)
```

## ADK agent config (agent.yaml)

If your agent uses `agent.yaml` for model config, set the env var in your run command:

```yaml
# agent.yaml — no changes needed here
model:
  provider: anthropic
  model_id: claude-sonnet-4-6
```

```bash
ANTHROPIC_BASE_URL=http://localhost:8899 adk run my_agent/
```

## Notes

- `ANTHROPIC_BASE_URL` is the standard override for the Anthropic Python SDK; ADK inherits it automatically
- Tool-heavy ADK agents benefit most from ContextPrune's tool schema filtering
- Stats logged to `~/.contextprune/stats.jsonl`

See real-world results in our benchmarks.
