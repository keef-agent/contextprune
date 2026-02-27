"""Semantic deduplication across messages using embedding-based similarity.

Detects and removes semantically redundant content across messages using
sentence-transformer embeddings. Chunks that are too similar to earlier chunks
get pruned, preserving earlier occurrences over later ones.

Default model: nomic-ai/nomic-embed-text-v1.5 (2048-token context window,
Apache 2.0 license). Falls back to all-MiniLM-L6-v2 (22MB, 256-token context).

Also provides MMRSelector for within-message chunk selection using Maximum
Marginal Relevance (Carbonell & Goldstein, SIGIR 1998).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_FALLBACK_MODEL = "all-MiniLM-L6-v2"

# nomic-embed-text-v1.5 requires task prefixes for optimal retrieval performance.
# See: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
_STORE_PREFIX = "search_document: "   # prefix for chunks stored in the seen pool
_QUERY_PREFIX = "search_query: "      # prefix for chunks being compared against pool


def _split_chunks(text: str, chunk_by: str = "sentence") -> List[str]:
    """Split text into chunks using the specified strategy.

    Args:
        text: Input text to split.
        chunk_by: Splitting strategy. One of:
            - "sentence": Split on sentence-ending punctuation (default).
            - "paragraph": Split on blank lines.
            - "chunk": Split on single newlines.

    Returns:
        List of non-empty chunk strings.
    """
    if chunk_by == "sentence":
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in parts if s.strip()]
    elif chunk_by == "paragraph":
        parts = re.split(r"\n\s*\n", text.strip())
        return [p.strip() for p in parts if p.strip()]
    elif chunk_by == "chunk":
        parts = re.split(r"\n+", text.strip())
        return [p.strip() for p in parts if p.strip()]
    else:
        raise ValueError(
            f"Unknown chunk_by: {chunk_by!r}. Use 'sentence', 'paragraph', or 'chunk'."
        )


def _count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 characters per token."""
    return max(1, len(text) // 4)


def _require_embeddings():
    """Import embeddings module with a helpful error on missing deps."""
    try:
        from . import embeddings
        return embeddings
    except ImportError:
        pass
    try:
        import contextprune.embeddings as embeddings
        return embeddings
    except ImportError:
        pass
    raise ImportError(
        "sentence-transformers is required for SemanticDeduplicator. "
        "Install it with: pip install sentence-transformers"
    )


def _is_instructional(text: str) -> bool:
    """Heuristic: return True if a chunk reads as an imperative instruction.

    Matches patterns like "Do not ...", "Never ...", "Always ...", "You must ...",
    "You are ...", "Do not ...", "NEVER ...", etc.  Used by protect_system mode to
    flag chunks that must be kept even if semantically similar to another chunk.

    This is intentionally broad — false positives (marking a non-instruction as
    instructional) are safe; false negatives (missing an instruction) are not.
    """
    imperative_prefixes = (
        r"^(do not|don't|never|always|must|you must|you are|you should|"
        r"you will|you cannot|you can't|do not|ensure|make sure|"
        r"remember|important|note:|warning:|critical:)"
    )
    return bool(re.match(imperative_prefixes, text.strip(), re.IGNORECASE))


class SemanticDeduplicator:
    """Remove semantically redundant content across messages using embeddings.

    Uses sentence-transformer embeddings to detect near-duplicate chunks
    across the entire message history. Earlier occurrences are kept; later
    duplicates are pruned.

    Algorithm:
        1. Split each message into chunks (sentences by default).
        2. Batch-embed all eligible chunks (one encode() call per message).
        3. For each chunk: compute cosine similarity against all previously-kept
           chunks across the entire conversation.
        4. If max similarity >= threshold: mark as duplicate and skip.
        5. If below threshold: keep and add to the seen pool.
        6. Reconstruct message content from kept chunks only.

    The default threshold (0.82) is a starting point. For research use, tune via
    grid search (0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95) on your validation set.

    Safety note on protect_system (default True):
        Research on KV cache compression ("The Pitfalls of KV Cache Compression",
        arXiv 2510.00231, 2025) showed that compression can silently cause LLMs to
        ignore certain instructions — particularly system-level rules that are
        phrased similarly to each other. With protect_system=True (the default),
        the system prompt is NEVER compressed: all its chunks are added to the
        seen pool so that duplicate content in *messages* is still stripped, but
        the system prompt itself is returned unchanged.

    Attributes:
        removal_log: List of (removed_chunk, best_matching_kept_chunk, score)
            tuples, populated after each call to deduplicate(). Useful for
            error analysis and threshold tuning.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.82,
        model: str = _DEFAULT_MODEL,
        chunk_by: str = "sentence",
        min_chunk_tokens: int = 5,
        preserve_first: bool = True,
        protect_system: bool = True,
        dedup_tool_results: bool = False,
    ) -> None:
        """Initialize the deduplicator.

        Args:
            similarity_threshold: Cosine similarity threshold above which a
                chunk is considered a duplicate. Range [0, 1]. Default: 0.82.
            model: HuggingFace model name. Default: nomic-ai/nomic-embed-text-v1.5.
                Use "minilm" or "all-MiniLM-L6-v2" for the fast 22MB fallback.
            chunk_by: Splitting strategy: "sentence" | "paragraph" | "chunk".
            min_chunk_tokens: Minimum estimated token count for a chunk to be
                eligible for deduplication (estimated as len(text)//4). Very
                short fragments like "OK.", "Sure.", single-word replies
                (< 5 estimated tokens / ~20 chars) are passed through unchanged.
                Default: 5.
            preserve_first: Always keep the first occurrence of any content
                (never removes the first chunk seen). Default: True.
            protect_system: If True (default), the system prompt is never
                compressed — all its chunks are added to the seen pool so
                message-level duplicates are still stripped, but the system
                prompt itself is returned byte-for-byte unchanged.

                Set to False only if you have measured that your system prompt
                contains genuine redundancy and you have validated that removing
                it does not affect task completion. See safety note in the class
                docstring.
            dedup_tool_results: If True, deduplicate content inside tool_result
                blocks (file reads, shell output, etc.) against the seen pool.
                Default: False.

                Tool results contain raw factual data — file contents, command
                output, API responses. Stripping sentences from them can give
                the model an incomplete or misleading view of the actual data,
                which can cause silent reasoning errors. Enable only if you have
                verified that your tool outputs genuinely repeat context already
                present elsewhere and that the model does not need to see the
                full output to complete the task.
        """
        # Normalize shorthand model names
        if model == "minilm":
            model = _FALLBACK_MODEL
        self.similarity_threshold = similarity_threshold
        self.model = model
        self.chunk_by = chunk_by
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_first = preserve_first
        self.protect_system = protect_system
        self.dedup_tool_results = dedup_tool_results
        self.removal_log: List[Tuple[str, str, float]] = []

    def deduplicate(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
        """Deduplicate messages and optional system prompt.

        Processes the system prompt first (if provided), then messages in order.
        Later duplicates are pruned in favor of earlier occurrences.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            system: Optional system prompt string.

        Returns:
            Tuple of (new_messages, new_system, sentences_removed):
                - new_messages: Messages with duplicate chunks removed.
                - new_system: System prompt with duplicate chunks removed (or
                  the original if unchanged / not provided).
                - sentences_removed: Total number of chunks removed.
        """
        emb = _require_embeddings()

        # Global pool of kept embeddings (document-prefixed, L2-normalized)
        seen_embeddings: List[np.ndarray] = []
        seen_texts: List[str] = []  # parallel to seen_embeddings for logging
        self.removal_log = []
        removed_count = 0

        def process_chunks(chunks: List[str]) -> List[str]:
            """Process chunks, returning only the non-duplicate ones."""
            nonlocal seen_embeddings, seen_texts, removed_count

            if not chunks:
                return chunks

            # Separate tiny chunks (pass through) from eligible (dedup-worthy) chunks
            eligible_indices: List[int] = []
            for i, chunk in enumerate(chunks):
                if _count_tokens_approx(chunk) >= self.min_chunk_tokens:
                    eligible_indices.append(i)

            if not eligible_indices:
                return chunks  # All too short — pass everything through

            eligible_texts = [chunks[i] for i in eligible_indices]

            # Batch embed with document prefix (for storage in the seen pool)
            doc_embs = emb.embed(eligible_texts, self.model, prefix=_STORE_PREFIX)
            # Batch embed with query prefix (for comparison against the seen pool)
            qry_embs = emb.embed(eligible_texts, self.model, prefix=_QUERY_PREFIX)

            remove_set: set = set()
            if seen_embeddings:
                # Vectorized similarity computation: compare all query embeddings
                # against all seen embeddings in one matrix multiplication
                seen_matrix = np.vstack(seen_embeddings)  # (n_seen, dim)
                qry_matrix = qry_embs  # (n_chunks, dim)
                # Dot product of normalized vectors = cosine similarity (batched)
                sim_matrix = qry_matrix @ seen_matrix.T  # (n_chunks, n_seen)

                # Process each chunk
                for local_idx, orig_idx in enumerate(eligible_indices):
                    chunk = chunks[orig_idx]
                    doc_emb = doc_embs[local_idx]
                    sims = sim_matrix[local_idx]
                    max_sim = float(sims.max())
                    best_match_idx = int(sims.argmax())

                    if max_sim >= self.similarity_threshold:
                        remove_set.add(orig_idx)
                        removed_count += 1
                        self.removal_log.append(
                            (chunk, seen_texts[best_match_idx], max_sim)
                        )
                    else:
                        seen_embeddings.append(doc_emb)
                        seen_texts.append(chunk)
            else:
                # First chunk ever — always keep (preserve_first semantics)
                # Add all eligible chunks from first message
                for local_idx, orig_idx in enumerate(eligible_indices):
                    seen_embeddings.append(doc_embs[local_idx])
                    seen_texts.append(chunks[orig_idx])

            # Reconstruct: keep tiny chunks + non-duplicate eligible chunks
            return [c for i, c in enumerate(chunks) if i not in remove_set]

        # --- Process system prompt first ---
        new_system = system
        if system and system.strip():
            sys_chunks = _split_chunks(system, self.chunk_by)
            if self.protect_system:
                # Safe mode: add all eligible system chunks to the seen pool
                # WITHOUT removing any of them.  The system prompt is returned
                # byte-for-byte unchanged so no instruction can be silently
                # dropped.  Message-level content that repeats the system prompt
                # is still stripped (because those chunks are now in the pool).
                eligible_sys = [
                    c for c in sys_chunks
                    if _count_tokens_approx(c) >= self.min_chunk_tokens
                ]
                if eligible_sys:
                    doc_embs = emb.embed(
                        eligible_sys, self.model, prefix=_STORE_PREFIX
                    )
                    for i, chunk in enumerate(eligible_sys):
                        seen_embeddings.append(doc_embs[i])
                        seen_texts.append(chunk)
                new_system = system  # unchanged
            else:
                # Opt-out: allow the system prompt to be deduplicated.
                # Process chunks one at a time so each chunk is checked
                # against previously-added chunks (enables within-system dedup).
                # Only use if you have validated that removing similar
                # instructions does not affect task completion.
                # See class docstring for the safety note.
                kept_sys: List[str] = []
                for chunk in sys_chunks:
                    result = process_chunks([chunk])
                    kept_sys.extend(result)
                new_system = " ".join(kept_sys) if kept_sys else system

        def dedup_text(text: str) -> str:
            """Dedup a plain text string, return kept text joined."""
            if not text.strip():
                return text
            chunks = _split_chunks(text, self.chunk_by)
            kept = process_chunks(chunks)
            return " ".join(kept) if kept else text

        def dedup_block_list(blocks: List[Any]) -> List[Any]:
            """Dedup text within a list of typed content blocks.

            Handles Anthropic content arrays: text blocks, tool_result blocks
            (which carry file/command output), and passes tool_use/image
            blocks through unchanged.
            """
            new_blocks: List[Any] = []
            for block in blocks:
                if not isinstance(block, dict):
                    new_blocks.append(block)
                    continue

                btype = block.get("type", "")

                if btype == "text":
                    raw = block.get("text", "")
                    deduped = dedup_text(raw)
                    new_block = dict(block)
                    new_block["text"] = deduped
                    new_blocks.append(new_block)

                elif btype == "tool_result":
                    # tool_result.content can be a string or a list of blocks.
                    # File reads, shell output, etc. end up here.
                    # Only dedup if explicitly enabled — stripping sentences from
                    # raw file/command output can silently corrupt the agent's
                    # view of the data.
                    if self.dedup_tool_results:
                        tc = block.get("content", "")
                        new_block = dict(block)
                        if isinstance(tc, str):
                            new_block["content"] = dedup_text(tc)
                        elif isinstance(tc, list):
                            new_block["content"] = dedup_block_list(tc)
                        new_blocks.append(new_block)
                    else:
                        # tool_result chunks are still added to the seen pool
                        # (so later messages don't repeat them), but the
                        # tool_result itself is forwarded unchanged.
                        tc = block.get("content", "")
                        if isinstance(tc, str) and tc.strip():
                            chunks = _split_chunks(tc, self.chunk_by)
                            process_chunks(chunks)  # seed the pool, discard output
                        elif isinstance(tc, list):
                            for inner in tc:
                                if isinstance(inner, dict) and inner.get("type") == "text":
                                    t = inner.get("text", "")
                                    if t.strip():
                                        process_chunks(_split_chunks(t, self.chunk_by))
                        new_blocks.append(block)

                else:
                    # tool_use (JSON input), image, etc. — pass through
                    new_blocks.append(block)

            return new_blocks

        # --- Process messages in order ---
        new_messages: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            new_msg = dict(msg)
            if isinstance(content, str):
                if content.strip():
                    new_msg["content"] = dedup_text(content)
            elif isinstance(content, list):
                # Anthropic-style typed block arrays (tool_use, tool_result, text)
                new_msg["content"] = dedup_block_list(content)
            new_messages.append(new_msg)

        return new_messages, new_system, removed_count


class MMRSelector:
    """Maximum Marginal Relevance chunk selector for within-message compression.

    Selects chunks that maximize relevance to the user query while minimizing
    redundancy with already-selected chunks. Produces grammatically coherent
    output by operating at the chunk level (not token level).

    Reference: Carbonell & Goldstein (1998). The use of MMR, diversity-based
    reranking for reordering documents and producing summaries. SIGIR.

    Advantage over LLMLingua-2: query-aware, coherent output, no ML model
    required beyond the embedding model already used for dedup.

    Parameters
    ----------
    token_budget_ratio : float
        Maximum fraction of tokens to *keep* (0–1). Default 0.5 keeps up to 50%
        of tokens, but actual reduction is driven by measured redundancy — if the
        content is not redundant the selector returns it unchanged.
    lambda_param : float
        MMR trade-off: 0 = pure diversity, 1 = pure relevance. Default 0.5.
    model : str
        HuggingFace embedding model name.
    chunk_by : str
        Splitting strategy: "paragraph" (default) | "sentence" | "chunk".
        Paragraph mode avoids cutting mid-sentence.
    min_chunk_tokens : int
        Chunks shorter than this (estimated token count) are always kept and
        excluded from MMR scoring. Default 30.
    preserve_order : bool
        If True (default), selected chunks are returned in their original
        document order rather than MMR selection order.
    redundancy_threshold : float
        Mean pairwise similarity above which MMR is applied. If the average
        chunk similarity is below this value the content is returned as-is
        (not redundant enough to compress). Default 0.35.
    min_tokens_to_mmr : int
        Only run MMR when total token count exceeds this value. Default 300.
    """

    def __init__(
        self,
        token_budget_ratio: float = 0.5,
        lambda_param: float = 0.5,
        model: str = _DEFAULT_MODEL,
        chunk_by: str = "paragraph",
        min_chunk_tokens: int = 30,
        preserve_order: bool = True,
        redundancy_threshold: float = 0.35,
        min_tokens_to_mmr: int = 300,
    ) -> None:
        if model == "minilm":
            model = _FALLBACK_MODEL
        self.token_budget_ratio = token_budget_ratio
        self.lambda_param = lambda_param
        self.model = model
        self.chunk_by = chunk_by
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_order = preserve_order
        self.redundancy_threshold = redundancy_threshold
        self.min_tokens_to_mmr = min_tokens_to_mmr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, content: str) -> List[str]:
        """Split content, falling back to sentence if no paragraphs found."""
        if self.chunk_by == "paragraph":
            chunks = _split_chunks(content, "paragraph")
            if len(chunks) <= 1:
                # No paragraph breaks — fall back to sentences
                chunks = _split_chunks(content, "sentence")
            return chunks
        return _split_chunks(content, self.chunk_by)

    @staticmethod
    def _count_tokens(chunks: List[str]) -> int:
        return sum(_count_tokens_approx(c) for c in chunks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        content: str,
        query: str,
        system: Optional[str] = None,  # noqa: ARG002 — reserved for future use
    ) -> Tuple[str, Dict[str, Any]]:
        """Select chunks using MMR to maximise relevance + diversity.

        Parameters
        ----------
        content : str
            The message content to compress.
        query : str
            The user's current question / task (used as the MMR query signal).
        system : str, optional
            Unused; reserved for future system-prompt-aware compression.

        Returns
        -------
        tuple[str, dict]
            (compressed_content, stats) where stats contains:
            - original_tokens  : int
            - selected_tokens  : int
            - reduction_pct    : float
            - chunks_total     : int
            - chunks_kept      : int
            - mmr_scores       : list[(chunk_prefix, score, "kept"|"dropped")]
        """
        emb = _require_embeddings()

        chunks = self._split(content)
        if not chunks:
            return content, self._passthrough_stats(content, 0, 0)

        # Separate tiny chunks (always kept, not MMR'd) from eligible chunks
        tiny: List[int] = []
        eligible: List[int] = []
        for i, c in enumerate(chunks):
            if _count_tokens_approx(c) < self.min_chunk_tokens:
                tiny.append(i)
            else:
                eligible.append(i)

        total_tokens = self._count_tokens(chunks)
        original_tokens = total_tokens

        # If content is too short or no eligible chunks — pass through unchanged
        if total_tokens < self.min_tokens_to_mmr or len(eligible) <= 1:
            stats = self._passthrough_stats(content, original_tokens, len(chunks))
            return content, stats

        eligible_texts = [chunks[i] for i in eligible]

        # Embed eligible chunks (document prefix) and query (query prefix)
        chunk_embs = emb.embed(eligible_texts, self.model, prefix=_STORE_PREFIX)
        query_embs = emb.embed([query], self.model, prefix=_QUERY_PREFIX)
        query_vec = query_embs[0]  # shape (dim,)

        # Check mean pairwise similarity to estimate redundancy
        if len(eligible) >= 2:
            # Fast batch pairwise: dot product of normalised vectors = cos sim
            sim_matrix = chunk_embs @ chunk_embs.T  # (n, n)
            # Upper triangle, excluding diagonal
            n = len(eligible)
            triu_vals = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
            mean_sim = float(np.mean(triu_vals)) if triu_vals else 0.0
        else:
            mean_sim = 0.0

        # Not enough redundancy — return unchanged
        if mean_sim < self.redundancy_threshold:
            stats = self._passthrough_stats(content, original_tokens, len(chunks))
            stats["mean_pairwise_similarity"] = round(mean_sim, 3)
            stats["skipped_reason"] = "low_redundancy"
            return content, stats

        # Compute per-chunk relevance scores against query
        relevance_scores = (chunk_embs @ query_vec).tolist()  # (n,)

        # Determine token budget for eligible chunks
        tiny_tokens = sum(_count_tokens_approx(chunks[i]) for i in tiny)
        eligible_tokens_total = total_tokens - tiny_tokens
        # Target: keep token_budget_ratio fraction of ELIGIBLE tokens
        # But don't cut more than (1 - min cap) of eligible content
        target_eligible_tokens = max(
            int(eligible_tokens_total * self.token_budget_ratio),
            self.min_chunk_tokens * 2,  # always keep at least ~2 chunks worth
        )

        # Greedy MMR selection
        remaining = list(range(len(eligible)))  # indices into eligible_texts
        selected: List[int] = []         # indices into eligible_texts
        selected_tokens = 0
        selected_embs: List[np.ndarray] = []

        mmr_log: List[Tuple[str, float, str]] = []

        while remaining and selected_tokens < target_eligible_tokens:
            if not selected:
                # First chunk: pick highest relevance
                scores = [relevance_scores[i] for i in remaining]
                best_local = int(np.argmax(scores))
                best_idx = remaining[best_local]
                mmr_score = relevance_scores[best_idx]
            else:
                # MMR score = λ * cos(chunk, query) - (1-λ) * max_cos(chunk, selected)
                sel_matrix = np.vstack(selected_embs)  # (k, dim)
                best_score = -float("inf")
                best_idx = remaining[0]
                for idx in remaining:
                    rel = self.lambda_param * relevance_scores[idx]
                    sims_to_selected = chunk_embs[idx] @ sel_matrix.T  # (k,)
                    div = (1 - self.lambda_param) * float(sims_to_selected.max())
                    score = rel - div
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                mmr_score = best_score

            chunk_tok = _count_tokens_approx(eligible_texts[best_idx])
            selected.append(best_idx)
            selected_tokens += chunk_tok
            selected_embs.append(chunk_embs[best_idx])
            remaining.remove(best_idx)
            mmr_log.append((eligible_texts[best_idx][:50], round(mmr_score, 4), "kept"))

        # Mark dropped chunks
        selected_set = set(selected)
        for idx in range(len(eligible)):
            if idx not in selected_set:
                mmr_log.append(
                    (eligible_texts[idx][:50], 0.0, "dropped")
                )

        # Reconstruct: tiny chunks always included; eligible only if selected
        if self.preserve_order:
            kept_original_indices = set(tiny) | {eligible[i] for i in selected}
            kept_chunks = [c for i, c in enumerate(chunks) if i in kept_original_indices]
        else:
            # Return in MMR selection order (tiny first, then selected in MMR order)
            kept_chunks = [chunks[i] for i in tiny] + [eligible_texts[i] for i in selected]

        # Join — use paragraph separator if chunk_by == paragraph, else space
        sep = "\n\n" if self.chunk_by == "paragraph" else " "
        compressed = sep.join(kept_chunks)

        selected_total_tokens = tiny_tokens + selected_tokens
        reduction_pct = round(
            max(0.0, (original_tokens - selected_total_tokens) / original_tokens * 100), 1
        )

        stats: Dict[str, Any] = {
            "original_tokens": original_tokens,
            "selected_tokens": selected_total_tokens,
            "reduction_pct": reduction_pct,
            "chunks_total": len(chunks),
            "chunks_kept": len(kept_chunks),
            "mmr_scores": mmr_log,
            "mean_pairwise_similarity": round(mean_sim, 3),
        }
        return compressed, stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _passthrough_stats(content: str, original_tokens: int, chunks_total: int) -> Dict[str, Any]:
        tok = original_tokens or _count_tokens_approx(content)
        return {
            "original_tokens": tok,
            "selected_tokens": tok,
            "reduction_pct": 0.0,
            "chunks_total": chunks_total,
            "chunks_kept": chunks_total,
            "mmr_scores": [],
        }
