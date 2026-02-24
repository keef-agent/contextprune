"""Lazy-loaded embedding model singleton. Shared across dedup and tool_filter.

Provides a cached model loader and helper functions for embedding text and
computing cosine similarity. Uses nomic-ai/nomic-embed-text-v1.5 by default
(2048-token context window, Apache 2.0) with all-MiniLM-L6-v2 as fallback.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np


@lru_cache(maxsize=2)
def get_model(model_name: str):
    """Load and cache an embedding model. First call ~2s, subsequent calls <1ms.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        A SentenceTransformer model instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        # trust_remote_code=True is required for nomic-embed-text-v1.5
        return SentenceTransformer(model_name, trust_remote_code=True)
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. "
            "Install it with: pip install sentence-transformers"
        )
    except Exception:
        # Fallback to MiniLM if the requested model fails to load
        if model_name != "all-MiniLM-L6-v2":
            return get_model("all-MiniLM-L6-v2")
        raise


def embed(texts: List[str], model_name: str, prefix: str = "") -> np.ndarray:
    """Embed a list of texts using the cached model.

    Args:
        texts: List of strings to embed.
        model_name: HuggingFace model name (passed to get_model).
        prefix: Optional prefix prepended to each text before encoding.
                Use "search_document: " for stored chunks and
                "search_query: " for query chunks (nomic-embed requirement).

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized.
    """
    model = get_model(model_name)
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(texts, normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized embedding vectors.

    Since both inputs are L2-normalized, cosine similarity reduces to dot product.

    Args:
        a: First normalized embedding vector.
        b: Second normalized embedding vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    return float(np.dot(a, b))
