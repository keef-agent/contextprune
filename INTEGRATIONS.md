# ContextPrune — Integration Quick Reference

Start the proxy first. Everything else is just pointing `base_url` at it.

```bash
contextprune serve --port 8899
```

---

## Real-world results (measured on live sessions)

- **OpenClaw 2-hour session:** 46% token reduction, 95 sentences removed
- **Agentic context (system + memory + tool outputs):** 36.6% reduction
- **Standard QA contexts with no redundancy:** 0% reduction (correctly passes through)

---

## OpenClaw

`~/.openclaw/openclaw.json`:

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

Restart OpenClaw.

---

## Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:8899 claude
```

Or permanently:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8899
claude
```

---

## LangChain

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6", base_url="http://localhost:8899")
```

---

## LangGraph

```python
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from typing import TypedDict

llm = ChatAnthropic(model="claude-sonnet-4-6", base_url="http://localhost:8899")

class State(TypedDict):
    messages: list
    result: str

def call_model(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"], "result": response.content}

graph = StateGraph(State)
graph.add_node("model", call_model)
graph.set_entry_point("model")
graph.add_edge("model", END)
app = graph.compile()
```

---

## OpenAI Agents SDK

```python
from agents import set_default_client, Agent, Runner
from openai import AsyncOpenAI

set_default_client(AsyncOpenAI(base_url="http://localhost:8899"))

agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    model="claude-sonnet-4-6",
)
```

---

## AG2 / AutoGen

```python
import autogen

config_list = [
    {
        "model": "claude-sonnet-4-6",
        "base_url": "http://localhost:8899",
        "api_key": "your-anthropic-api-key",
        "api_type": "anthropic",
    }
]

llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent(name="assistant", llm_config=llm_config)
```

---

## CrewAI

```python
from crewai import Agent, Task, Crew
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6", base_url="http://localhost:8899")

agent = Agent(
    role="Researcher",
    goal="Gather and summarize information",
    backstory="Expert researcher.",
    llm=llm,
)
```

---

## PydanticAI

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel("claude-sonnet-4-6", base_url="http://localhost:8899")
agent = Agent(model)

result = agent.run_sync("What are the three laws of thermodynamics?")
print(result.data)
```

---

## Google ADK

```bash
export ANTHROPIC_BASE_URL=http://localhost:8899
adk run my_agent/
```

Or inline:

```bash
ANTHROPIC_BASE_URL=http://localhost:8899 adk run my_agent/
```

---

## Mastra (TypeScript)

```typescript
import { Mastra } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({ baseURL: 'http://localhost:8899' });

const agent = new Agent({
  name: 'my-agent',
  instructions: 'You are a helpful assistant.',
  model: { provider: 'ANTHROPIC', name: 'claude-sonnet-4-6', toolChoice: 'auto' },
  client: anthropic,
});
```

---

## Vercel AI SDK (TypeScript)

```typescript
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText } from 'ai';

const anthropic = createAnthropic({ baseURL: 'http://localhost:8899' });

const { text } = await generateText({
  model: anthropic('claude-sonnet-4-6'),
  prompt: 'Explain gradient descent.',
});
```

---

## NanoClaw

```bash
export ANTHROPIC_BASE_URL=http://localhost:8899
nanoclaw start
```

Or inline:

```bash
ANTHROPIC_BASE_URL=http://localhost:8899 nanoclaw start
```

---

## LlamaIndex

```python
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings

llm = Anthropic(model="claude-sonnet-4-6", base_url="http://localhost:8899")
Settings.llm = llm

response = llm.complete("Explain the CAP theorem in two sentences.")
print(response.text)
```

---

## Verify any integration

```bash
# Stats log — one line per compressed request
tail -f ~/.contextprune/stats.jsonl

# Dry-run test (no API key needed)
curl -X POST http://localhost:8899/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 10,
    "system": "You are helpful. You are helpful. You are helpful.",
    "messages": [{"role": "user", "content": "Say hi."}],
    "__contextprune_dry_run": true
  }'
```

Full integration guides: [docs/integrations/](docs/integrations/)
