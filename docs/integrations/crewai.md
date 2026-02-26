# ContextPrune + CrewAI

CrewAI uses LangChain's `ChatAnthropic` under the hood. Pass `base_url` when constructing the LLM and assign it to each agent.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
pip install crewai langchain-anthropic
```

## Integration

```python
from crewai import Agent, Task, Crew
from langchain_anthropic import ChatAnthropic

# All agent calls route through ContextPrune
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    base_url="http://localhost:8899",
    # api_key is read from ANTHROPIC_API_KEY env var
)

researcher = Agent(
    role="Research Analyst",
    goal="Find and summarize key facts on a given topic",
    backstory="You are an expert researcher with 10 years of experience.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Turn research summaries into clear, concise reports",
    backstory="You write clear documentation for technical audiences.",
    llm=llm,
    verbose=True,
)

research_task = Task(
    description="Research the latest developments in transformer architecture (2024-2025).",
    agent=researcher,
    expected_output="A bullet-point summary of the top 5 developments.",
)

write_task = Task(
    description="Turn the research summary into a 2-paragraph technical report.",
    agent=writer,
    expected_output="A polished 2-paragraph report suitable for a technical blog.",
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True,
)

result = crew.kickoff()
print(result)
```

## Notes

- CrewAI agent `backstory`, `goal`, and task context accumulate across the crew run — ideal for deduplication
- Use the same `llm` instance across all agents so ContextPrune sees the full repeated context
- Stats logged to `~/.contextprune/stats.jsonl`
- Verify `langchain-anthropic >= 0.1.0` for `base_url` support

See real-world results in our benchmarks.
