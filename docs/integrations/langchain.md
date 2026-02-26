# ContextPrune + LangChain / LangGraph

Route LangChain and LangGraph Anthropic calls through the ContextPrune proxy. No changes to your chain logic — just point `base_url` at the proxy.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
pip install langchain-anthropic langchain-core
```

## LangChain

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    base_url="http://localhost:8899",
    # api_key is read from ANTHROPIC_API_KEY env var
)

response = llm.invoke("Summarize the key points of the Gettysburg Address.")
print(response.content)
```

## LangGraph

```python
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from typing import TypedDict

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    base_url="http://localhost:8899",
)

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

result = app.invoke({"messages": [{"role": "user", "content": "What is 2+2?"}]})
print(result["result"])
```

## LangChain with tools

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    base_url="http://localhost:8899",
)

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Boston?")
print(response.tool_calls)
```

## Notes

- `base_url` is supported in `langchain-anthropic >= 0.1.0`
- Streaming is passed through to Anthropic unchanged; deduplication runs on non-streaming requests
- ContextPrune logs savings to `~/.contextprune/stats.jsonl`

See real-world results in our benchmarks.
