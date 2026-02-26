"""contextprune -- Cut your LLM API costs by 40-80% with 2 lines of code."""

from .core import Config, wrap, wrap_openai, WrappedClient, WrappedOpenAIClient
from .stats import CompressionStats


def serve(*args, **kwargs):
    """Start the ContextPrune proxy server. See contextprune.proxy.serve for args."""
    from .proxy import serve as _serve  # lazy import to avoid circular-import RuntimeWarning
    return _serve(*args, **kwargs)


__all__ = [
    "wrap",
    "wrap_openai",
    "Config",
    "CompressionStats",
    "WrappedClient",
    "WrappedOpenAIClient",
    "serve",
]
