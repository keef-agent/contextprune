"""Basic usage example for contextprune.

Shows how to wrap an Anthropic client and check compression stats.
"""

import anthropic
from contextprune import wrap, Config

# Wrap the Anthropic client
client = wrap(
    anthropic.Anthropic(),
    config=Config(verbose=True),  # Print savings per call
)

# Use it exactly like normal
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a helpful coding assistant. You write clean, well-tested Python code.",
    messages=[
        {"role": "user", "content": "Write a function to check if a number is prime."},
    ],
)

print(response.content[0].text)
print(f"\nCompression stats: {response.compression_stats}")
