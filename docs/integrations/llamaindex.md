# ContextPrune + LlamaIndex

LlamaIndex's Anthropic LLM wrapper accepts a `base_url` parameter. Point it at the ContextPrune proxy.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
pip install llama-index llama-index-llms-anthropic
```

## Integration

```python
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings

llm = Anthropic(
    model="claude-sonnet-4-6",
    base_url="http://localhost:8899",
    # api_key is read from ANTHROPIC_API_KEY env var
)

# Set as global default
Settings.llm = llm

# Direct completion
response = llm.complete("Explain the CAP theorem in two sentences.")
print(response.text)
```

## With a query engine

```python
from llama_index.llms.anthropic import Anthropic
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

llm = Anthropic(model="claude-sonnet-4-6", base_url="http://localhost:8899")
Settings.llm = llm

# Load documents and build index
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query — context-heavy RAG calls benefit from deduplication
query_engine = index.as_query_engine()
response = query_engine.query("What are the main themes in the documentation?")
print(response)
```

## Chat with history

```python
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import ChatMessage, MessageRole

llm = Anthropic(model="claude-sonnet-4-6", base_url="http://localhost:8899")

messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a concise technical assistant."),
    ChatMessage(role=MessageRole.USER, content="What is attention in transformers?"),
]

response = llm.chat(messages)
print(response.message.content)
```

## Async

```python
import asyncio
from llama_index.llms.anthropic import Anthropic

llm = Anthropic(model="claude-sonnet-4-6", base_url="http://localhost:8899")

async def main():
    response = await llm.acomplete("List three properties of a good vector database.")
    print(response.text)

asyncio.run(main())
```

## Notes

- `base_url` supported in `llama-index-llms-anthropic >= 0.1.0`
- RAG pipelines with many retrieved chunks benefit when documents overlap; unique content passes through at 0% overhead
- Stats logged to `~/.contextprune/stats.jsonl`

See real-world results in our benchmarks.
