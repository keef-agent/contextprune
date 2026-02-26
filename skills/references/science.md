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

## Papers

### LLMLingua (Jiang et al., 2023)
**Paper:** https://arxiv.org/abs/2310.05736

Demonstrated 20x context compression with <1.5% accuracy loss using token-level importance scoring (perplexity from a small LM). ContextPrune uses sentence-level semantic similarity instead — safer for agent contexts where removing individual tokens can break reasoning chains and cause hallucination.

### ACON (Zhong et al., 2024)
**Paper:** https://arxiv.org/abs/2406.06548

The most directly relevant prior work. Showed **26-54% context reduction** in agentic workloads using semantic redundancy detection — specifically targeting the agent memory + tool output + history overlap that ContextPrune addresses. Validates both the approach and the magnitude of reduction achievable.

### Token Elasticity / TALE (Ivgi et al., 2024)
**Paper:** https://arxiv.org/abs/2407.07955

Quantified that LLMs require explicit, calibrated token budgets. Vague instructions like "be concise" or "summarize briefly" don't reliably reduce output length. ContextPrune bypasses this problem entirely by removing redundancy at the infrastructure level before the model ever sees the context — no instruction-following required.

## Real-World Results

Measured on live sessions (not synthetic benchmarks):

| Context type | Reduction |
|---|---|
| 2-hour OpenClaw agent session | **46%** |
| Agentic context (system + memory + tools) | **36-45%** |
| Non-redundant context | **0%** (guard fires correctly) |

The 46% result was measured on a keef-direct agent session with a large system prompt, multiple memory files, and accumulated tool outputs — exactly the workload ACON targets.
