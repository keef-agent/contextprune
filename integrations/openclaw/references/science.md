# ContextPrune: Science Reference

## What Semantic Deduplication Is

Semantic deduplication detects and removes sentences that express the same meaning, even when worded differently. Unlike exact-match deduplication (which only catches copy-paste repetition), semantic dedup catches paraphrases, restatements, and summaries of information already present in the context.

This is the dominant source of waste in agentic LLM contexts: system prompts, memory files, tool outputs, and conversation history all restate the same facts in different words. The model gets charged for all of them.

## The Embedding Model

**Model:** `paraphrase-MiniLM-L6-v2`
- 22M parameters
- Runs entirely locally — no external API calls
- Encodes a sentence in <50ms on CPU
- 384-dimensional embeddings
- Trained specifically for semantic similarity / paraphrase detection

Every sentence in the context is encoded into a vector. Pairwise cosine similarity is computed across all sentences. Sentences above the similarity threshold (default: 0.82) that are duplicates of earlier sentences are removed.

## Redundancy Guard

Before deduplication runs, ContextPrune checks mean pairwise similarity across the context. If it's below **0.35**, the context is not redundant enough to benefit — the request passes through unchanged. This is why non-redundant contexts see **0% reduction**: the guard fires and nothing is removed.

## Latency Overhead

ContextPrune adds processing time before forwarding each request. The cost is:

1. **Sentence splitting** — negligible (<5ms)
2. **Batch encoding** via `paraphrase-MiniLM-L6-v2` — the dominant cost
3. **Pairwise cosine similarity** — matrix multiply, negligible

Measured overhead on CPU:

| Context size | Sentences | Overhead |
|---|---|---|
| Small (system prompt only) | ~20-50 | ~50-150ms |
| Medium (system + memory files) | ~100-200 | ~200-500ms |
| Large (full agentic context) | ~300-500 | ~500ms-1s |

On GPU (CUDA available), all of the above drop by ~5-10x.

**Tradeoff:** For a large agentic context, you pay ~500ms-1s upfront to remove 40-50% of tokens. The model processes fewer tokens, which reduces TTFT and total generation time — partially or fully recovering the overhead depending on context size and model latency.

The redundancy guard short-circuits this cost: if mean pairwise similarity is below 0.35, encoding stops early and the request passes through immediately.

## Papers

### LLMLingua (Jiang et al., 2023)
**Paper:** https://arxiv.org/abs/2310.05736

Demonstrated 20x context compression with <1.5% accuracy loss using token-level importance scoring (perplexity from a small LM). ContextPrune uses sentence-level semantic similarity instead — safer for agent contexts where removing individual tokens can break reasoning chains and cause hallucination.

### ACON (Kang et al., Microsoft, 2025)
**Paper:** https://arxiv.org/abs/2510.00615

The most relevant prior work for validating ContextPrune's reduction targets. ACON used guideline optimization via failure analysis — different from ContextPrune's embedding approach — but independently confirmed that agent memory + tool output + history overlap is the dominant token waste in agentic workloads, and that **26-54% reduction** is achievable without accuracy loss. Validates the problem magnitude. ContextPrune targets the same redundancy via embedding-based deduplication instead of learned guidelines.

### Token Elasticity / TALE (Han et al., 2024)
**Paper:** https://arxiv.org/abs/2412.18547

Quantified that LLMs require explicit, calibrated token budgets. Vague instructions like "be concise" or "summarize briefly" don't reliably reduce output length. ContextPrune bypasses this problem entirely by removing redundancy at the infrastructure level before the model ever sees the context — no instruction-following required.

### The Pitfalls of KV Cache Compression (Valvoda et al., 2025)
**Paper:** https://arxiv.org/abs/2510.00231

Showed that context compression can cause LLMs to silently ignore certain instructions — particularly system-level rules that are phrased similarly to each other (e.g., "Do not share user data" and "Never reveal personal information"). The model doesn't error; it just stops following the rule. This effect is strongest for instructions that appear later in the context or that have high semantic overlap with other instructions.

ContextPrune addresses this directly via `protect_system=True` (the default): the system prompt is returned byte-for-byte unchanged, while all its chunks are added to the deduplication pool so that redundant content in *messages* is still stripped. Set `protect_system=False` only if you have validated that removing similar instructions from your system prompt does not affect task completion.

## Real-World Results

Measured on live sessions (not synthetic benchmarks):

| Context type | Reduction |
|---|---|
| 2-hour agentic session (large system prompt + memory files + tool outputs) | **46%** |
| Non-redundant context | **0%** (redundancy guard fires correctly) |

The 46% result was measured on a live OpenClaw agent session running Claude Sonnet 4.6, with a large system prompt (~8K tokens), multiple injected memory files, and accumulated tool outputs across a 2-hour run. Token counts before and after deduplication were recorded per-request via `stats.jsonl`.
