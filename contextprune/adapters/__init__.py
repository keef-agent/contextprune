"""Multi-model adapter layer for contextprune benchmarks."""

from .openrouter import CompletionResult, OpenRouterAdapter, SUPPORTED_MODELS

__all__ = ["OpenRouterAdapter", "CompletionResult", "SUPPORTED_MODELS"]
