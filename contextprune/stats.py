"""Compression statistics tracking."""

from __future__ import annotations

import dataclasses
import time
from typing import Optional


@dataclasses.dataclass
class CompressionStats:
    """Stats from a single compression pass."""

    original_tokens: int = 0
    compressed_tokens: int = 0
    savings_pct: float = 0.0
    time_ms: float = 0.0
    dedup_removed: int = 0
    tools_removed: int = 0
    budget_injected: bool = False

    def __repr__(self) -> str:
        return (
            f"CompressionStats("
            f"original_tokens={self.original_tokens}, "
            f"compressed_tokens={self.compressed_tokens}, "
            f"savings_pct={self.savings_pct}, "
            f"time_ms={self.time_ms})"
        )


class StatsTimer:
    """Context manager for timing compression passes."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "StatsTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if self._start is not None:
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000
