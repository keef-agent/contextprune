"""Baselines for head-to-head comparison with ContextPrune.

Available baselines:
    LLMLingua2Baseline — Microsoft's LLMLingua-2 (ACL Findings 2024)

Usage:
    from contextprune.baselines import LLMLingua2Baseline

    baseline = LLMLingua2Baseline(rate=0.5)
    compressed_msgs, compressed_sys, stats = baseline.compress(messages, system)
"""

from .llmlingua2 import LLMLingua2Baseline

__all__ = ["LLMLingua2Baseline"]
