# ContextPrune + AG2 / AutoGen

AutoGen (AG2) accepts `base_url` in its `config_list`. Swap the Anthropic endpoint for the ContextPrune proxy URL.

## Setup

**1. Start the proxy**

```bash
contextprune serve --port 8899
```

**2. Install dependencies**

```bash
pip install pyautogen
# or for AG2:
pip install ag2
```

## Integration

```python
import autogen

config_list = [
    {
        "model": "claude-sonnet-4-6",
        "base_url": "http://localhost:8899",
        "api_key": "your-anthropic-api-key",  # forwarded as-is to Anthropic
        "api_type": "anthropic",
    }
]

llm_config = {
    "config_list": config_list,
    "cache_seed": None,
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant.",
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config={"work_dir": "/tmp/autogen", "use_docker": False},
)

user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to compute the nth Fibonacci number.",
)
```

## Multi-agent setup

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

planner = autogen.AssistantAgent(
    name="planner",
    llm_config=llm_config,
    system_message="Break down complex tasks into steps.",
)

executor = autogen.AssistantAgent(
    name="executor",
    llm_config=llm_config,
    system_message="Implement the plan provided by the planner.",
)

user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
)

# Start a group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, executor],
    messages=[],
    max_round=6,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(manager, message="Build a web scraper for Hacker News.")
```

## Notes

- Both `autogen` (Microsoft) and `ag2` (fork) support `base_url` in `config_list`
- Multi-agent setups benefit most — repeated system prompts and message history across many turns compress well
- Stats logged to `~/.contextprune/stats.jsonl`

See real-world results in our benchmarks.
