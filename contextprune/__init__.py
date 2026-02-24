"""contextprune -- Cut your LLM API costs by 40-80% with 2 lines of code."""

from .core import Config, wrap, wrap_openai, WrappedClient, WrappedOpenAIClient
from .stats import CompressionStats

__all__ = [
    "wrap",
    "wrap_openai",
    "Config",
    "CompressionStats",
    "WrappedClient",
    "WrappedOpenAIClient",
]
