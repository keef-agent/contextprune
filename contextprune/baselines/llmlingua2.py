"""LLMLingua-2 baseline wrapper for head-to-head comparison with ContextPrune.

Paper: LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic
       Prompt Compression (Pan et al., ACL Findings 2024)
       https://arxiv.org/abs/2403.12968

Model: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank

Wraps the llmlingua library to match ContextPrune's message-list interface,
enabling direct comparison in benchmark experiments.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def _count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 characters per token (tiktoken-consistent approx)."""
    return max(0, len(text) // 4)


@lru_cache(maxsize=4)
def _get_compressor(model_name: str, device: str):
    """Load and cache a PromptCompressor instance.

    First call downloads the model (~500MB) and takes 10-30s.
    Subsequent calls return the cached instance in <1ms.

    Args:
        model_name: HuggingFace model name for LLMLingua-2.
        device: Device map string ('cpu', 'cuda', 'mps').

    Returns:
        A llmlingua.PromptCompressor instance.

    Raises:
        ImportError: If llmlingua is not installed.
    """
    try:
        from llmlingua import PromptCompressor
    except ImportError:
        raise ImportError("Install llmlingua: pip install llmlingua")

    logger.info(
        "Loading LLMLingua-2 model %s on %s (first call may take 10-30s)...",
        model_name,
        device,
    )
    compressor = PromptCompressor(
        model_name=model_name,
        use_llmlingua2=True,
        device_map=device,
    )
    logger.info("LLMLingua-2 model loaded.")
    return compressor


def _extract_content(msg: Dict[str, Any]) -> str:
    """Extract text content from a message dict (handles str and list content)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return " ".join(parts)
    return ""


class LLMLingua2Baseline:
    """LLMLingua-2 baseline wrapper for head-to-head comparison with ContextPrune.

    Paper: LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic
           Prompt Compression (Pan et al., ACL Findings 2024)
    Model: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank

    Wraps the llmlingua library to match ContextPrune's message-list interface,
    enabling direct comparison in benchmark experiments.

    Usage:
        baseline = LLMLingua2Baseline(rate=0.5)
        compressed_msgs, compressed_sys, stats = baseline.compress(messages, system)
        print(stats)
        # {'original_tokens': 1248, 'compressed_tokens': 624, 'reduction_pct': 50.0,
        #  'method': 'llmlingua2', 'rate': 0.5}
    """

    def __init__(
        self,
        rate: float = 0.5,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        force_tokens: Optional[List[str]] = None,
    ) -> None:
        """Initialize the LLMLingua-2 baseline.

        Args:
            rate: Target compression ratio. 0.5 = keep 50% of tokens.
                  Range (0, 1]. Passed directly to llmlingua.
            model_name: HuggingFace model name. Default is the ACL 2024 paper model.
            device: Device for model inference. 'cpu', 'cuda', or 'mps'.
            force_tokens: Tokens that must be preserved during compression.
                          Defaults to ['\\n'] to preserve newlines.
        """
        if not (0 < rate <= 1):
            raise ValueError(f"rate must be in (0, 1], got {rate}")
        self.rate = rate
        self.model_name = model_name
        self.device = device
        self.force_tokens = force_tokens if force_tokens is not None else ["\n"]

    def _load_compressor(self):
        """Lazy-load and cache the PromptCompressor model."""
        return _get_compressor(self.model_name, self.device)

    def _compress_text(self, text: str) -> Tuple[str, int, int]:
        """Compress a single text string via LLMLingua-2.

        Args:
            text: Input text to compress.

        Returns:
            Tuple of (compressed_text, origin_tokens, compressed_tokens).
            Returns (text, approx_tokens, approx_tokens) if text is empty.
        """
        if not text or not text.strip():
            approx = _count_tokens_approx(text)
            return text, approx, approx

        compressor = self._load_compressor()
        result = compressor.compress_prompt(
            text,
            rate=self.rate,
            force_tokens=self.force_tokens,
        )

        compressed = result.get("compressed_prompt", text)
        origin_tokens = result.get("origin_tokens", _count_tokens_approx(text))
        compressed_tokens = result.get("compressed_tokens", _count_tokens_approx(compressed))

        logger.debug(
            "LLMLingua-2 compressed %d chars → %d chars | tokens %d → %d | "
            "rate requested=%.2f achieved=%.2f",
            len(text),
            len(compressed),
            origin_tokens,
            compressed_tokens,
            self.rate,
            (compressed_tokens / origin_tokens) if origin_tokens else 1.0,
        )

        return compressed, origin_tokens, compressed_tokens

    def _redistribute_compressed(
        self,
        original_messages: List[Dict[str, Any]],
        compressed_text: str,
    ) -> List[Dict[str, Any]]:
        """Redistribute compressed text back into message-dict structure.

        LLMLingua-2 operates on raw text. After compression, we split the
        compressed text proportionally by the original message lengths and
        assign the pieces back to their roles.

        This approach preserves role structure (user/assistant) while
        distributing the compressed content proportionally.

        Args:
            original_messages: Original list of message dicts with roles.
            compressed_text: The raw compressed text from LLMLingua-2.

        Returns:
            List of message dicts with roles intact and compressed content.
        """
        if not original_messages:
            return []

        # Compute original char lengths per message for proportional redistribution
        orig_contents = [_extract_content(m) for m in original_messages]
        orig_lengths = [len(c) for c in orig_contents]
        total_orig = sum(orig_lengths)

        if total_orig == 0:
            return list(original_messages)

        # Split compressed text proportionally by original message sizes
        compressed_chars = len(compressed_text)
        new_messages = []
        cursor = 0

        for i, msg in enumerate(original_messages):
            if i == len(original_messages) - 1:
                # Last message gets the remainder to avoid off-by-one splits
                piece = compressed_text[cursor:]
            else:
                fraction = orig_lengths[i] / total_orig
                piece_len = int(compressed_chars * fraction)
                piece = compressed_text[cursor : cursor + piece_len]
                cursor += piece_len

            new_msg = dict(msg)
            # Preserve non-string content (tool results, images) as-is
            if isinstance(msg.get("content"), str):
                new_msg["content"] = piece.strip() if piece.strip() else msg.get("content", "")
            new_messages.append(new_msg)

        return new_messages

    def compress(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        """Compress messages (and optionally a system prompt) using LLMLingua-2.

        LLMLingua-2 operates on raw text. This method:
          1. Compresses the system prompt separately (usually the largest chunk).
          2. Concatenates all message content and compresses in one pass.
          3. Redistributes compressed content back into the message-dict structure.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            system: Optional system prompt string to compress separately.

        Returns:
            Tuple of (compressed_messages, compressed_system, stats) where:
              - compressed_messages: Messages with compressed content, roles intact.
              - compressed_system: Compressed system prompt (or None if not provided).
              - stats: Dict with compression statistics:
                  {
                    "original_tokens": int,
                    "compressed_tokens": int,
                    "reduction_pct": float,
                    "method": "llmlingua2",
                    "rate": float,
                  }
        """
        total_origin = 0
        total_compressed = 0

        # --- Compress system prompt ---
        compressed_system = system
        if system and system.strip():
            comp_sys, sys_orig, sys_comp = self._compress_text(system)
            compressed_system = comp_sys
            total_origin += sys_orig
            total_compressed += sys_comp
            logger.info(
                "System prompt: %d → %d tokens (%.1f%% kept)",
                sys_orig,
                sys_comp,
                (sys_comp / sys_orig * 100) if sys_orig else 100,
            )

        # --- Compress messages ---
        compressed_messages = list(messages)
        if messages:
            # Concatenate all string message contents for a single compress call
            msg_texts = []
            non_text_indices = {}
            for i, msg in enumerate(messages):
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg_texts.append(content)
                else:
                    # Non-string content (images, tool results) — skip compression
                    non_text_indices[i] = msg
                    msg_texts.append("")  # placeholder

            combined_text = "\n\n".join(msg_texts)

            if combined_text.strip():
                comp_msgs, msgs_orig, msgs_comp = self._compress_text(combined_text)
                total_origin += msgs_orig
                total_compressed += msgs_comp

                # Only redistribute if we actually have text-bearing messages
                text_messages = [m for m in messages if isinstance(m.get("content"), str)]
                if text_messages:
                    redistributed = self._redistribute_compressed(text_messages, comp_msgs)

                    # Merge back: non-text messages pass through unchanged
                    compressed_messages = []
                    redi = 0
                    for i, msg in enumerate(messages):
                        if i in non_text_indices:
                            compressed_messages.append(msg)
                        else:
                            if redi < len(redistributed):
                                compressed_messages.append(redistributed[redi])
                                redi += 1
                            else:
                                compressed_messages.append(msg)

                logger.info(
                    "Messages: %d → %d tokens (%.1f%% kept)",
                    msgs_orig,
                    msgs_comp,
                    (msgs_comp / msgs_orig * 100) if msgs_orig else 100,
                )

        # --- Build stats ---
        reduction = (
            (total_origin - total_compressed) / total_origin * 100
            if total_origin > 0
            else 0.0
        )
        stats: Dict[str, Any] = {
            "original_tokens": total_origin,
            "compressed_tokens": total_compressed,
            "reduction_pct": round(max(0.0, reduction), 1),
            "method": "llmlingua2",
            "rate": self.rate,
        }

        logger.info(
            "LLMLingua-2 total: %d → %d tokens (%.1f%% reduction, rate=%.2f)",
            total_origin,
            total_compressed,
            reduction,
            self.rate,
        )

        return compressed_messages, compressed_system, stats
