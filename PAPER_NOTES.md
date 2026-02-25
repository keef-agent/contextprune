# ContextPrune — Paper Writing Notes

Live working document. Updated as findings come in. Use this when drafting sections.

---

## Dataset: CompressionSafety (Exp 0)

### Methodology decisions (for Section 4 writeup)

- **Source**: ShareGPT_Vicuna_unfiltered (HuggingFace: `anon8231489123/ShareGPT_Vicuna_unfiltered`)
- **Selection criteria**: conversations ≥ 1000 tokens (len(text)//4), randomly shuffled with seed=42
- **Annotation model**: GPT-5.2 Pro (web UI) — chosen for strongest instruction-following and reasoning
- **Second annotator**: Claude Sonnet 4.6 via OpenRouter — different provider, different training, same prompt
- **Sentences per conversation**: capped at 25 (truncated from end to preserve final question context)
- **Annotation prompt design**: sentence-level ternary labels (essential / redundant / uncertain), conditioned on the final user question. Annotator sees full conversation + final question before labeling.
- **Why ShareGPT**: largest publicly available real human-AI conversation dataset. Covers diverse domains. GPT-generated responses reflect real production LLM outputs.
- **Why 100 conversations**: at observed std=0.30, n=100 gives 95% CI ±5.9% on mean SCR. Sufficient for section-level statistical claims.

### Preliminary results (n=55, as of 2026-02-24)

| Metric | Value |
|--------|-------|
| n | 55 |
| Mean SCR | 60.6% |
| Median SCR | 68.0% |
| Std | 30.0% |
| 95% CI on mean | ±7.9% |
| Min | 0.0% (exercise instructions, RPG game rules) |
| Max | 100.0% (3 conversations: socket API, scientific review, thermodynamics) |

**Distribution (preliminary):**
| SCR Bucket | n | % | Interpretation |
|---|---|---|---|
| 0–20% | 9 | 16% | Dense procedural/code — nothing safe to cut |
| 20–40% | 6 | 11% | Structured technical — some repeated elaboration |
| 40–60% | 9 | 16% | Mixed task — moderate redundancy |
| 60–80% | 15 | 27% | Iterative task conversations — significant redundancy |
| 80–100% | 16 | 29% | Brainstorming/discarded context — almost all redundant |

**Key empirical observations so far:**

1. **Bimodal distribution**: two natural clusters — low-SCR (dense technical/procedural) and high-SCR (open-ended/iterative). Middle range under-represented. This is NOT a normal distribution.

2. **Domain-dependent SCR**: domain predicts SCR better than conversation length alone.
   - Exercise instructions, game rules, dense code: 0–8% (nothing to cut)
   - Legal motions, architecture specs, physics problems: 20–44%
   - Mixed coding sessions: 36–68%
   - Scientific literature review with tangential context: 68–96%
   - ASCII art games, tic-tac-toe rewrites, thermodynamics recaps: 100%

3. **100% redundant pattern**: occurs when the ENTIRE prior conversation is answering a DIFFERENT question than the final turn. The model accumulated context around a topic that changed.

4. **0% redundant pattern**: occurs with procedural/instructional content where every sentence is a step, constraint, or fact that the model needs. Creative fiction continuations also approach 0% — the story IS the content.

5. **Implication for ContextPrune**: a single global compression threshold is wrong. The system needs domain classification before deciding how aggressively to compress. This is why the domain-aware TALE budget classifier exists.

### Pending (will update at n=100)

- Final mean/median/std
- 95% CI (expected ±5.9%)
- IAA with Claude Sonnet 4.6 (Cohen's Kappa)
- Functional validation (compression-then-test answer preservation rate)
- Human spot-check agreement rate (Keith)

---

## Section 4 draft notes: CompressionSafety Dataset

**Opening claim to defend**: "The safe compression rate of real agent conversations ranges from 0% to 100% and is strongly predicted by conversation domain, not length."

**Table 4.1**: SCR distribution by bucket (use final n=100 numbers)

**Figure 4.1**: Histogram of SCR values across 100 conversations. Expected shape: bimodal with peaks at 0-20% and 80-100%.

**Table 4.2**: IAA results — Cohen's Kappa between GPT-5.2 and Claude Sonnet 4.6 annotations

**Key claim for abstract**: "We find that X% ± Y% of tokens in real agent conversations are safe to remove without changing model outputs, with substantial variation by domain (range: Z1%–Z2% mean by category)."

**Framing for reviewers**: No prior compression paper has measured ground-truth token redundancy. They all measure post-hoc compression ratio or task accuracy. This is the first dataset providing causal annotation (would removing this change the answer?).

---

## Section 5 draft notes: ContextPrune System

**Still needed before writing**: Exp 3 results (accuracy × compression × model)

**Known from synthetic benchmarks (v0.1, NOT for citation)**:
- Tool-heavy scenarios: 45% compression (ContextPrune) vs 3% (LLMLingua-2)
- Code agent: 42% vs 40%
- RAG context: 39% vs 47%
- Do NOT cite these — synthetic data

**Real numbers needed**: Exp 3 on MMLU-Pro, MATH-500, LiveCodeBench, FRAMES (post OpenRouter credits)

---

## Methodology decisions log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | nomic-embed-text-v1.5 | 2048-token window vs MiniLM's 256; MTEB 62 vs 57 |
| Dedup approach | MMR (greedy) | Query-aware; preserves grammatical coherence vs LLMLingua-2's token-dropping |
| Budget injection format | Natural language at end of system prompt | "Target: N tokens or fewer" — matches TALE paper's findings |
| Minimum budget | 50 tokens | Token Elasticity safety floor — below this, models use MORE tokens |
| LLMLingua-2 positioning | Complementary, not competing | ContextPrune wins tool-heavy (45% vs 3%); LLMLingua-2 wins dense text (47% vs 21%) |
| Benchmark replacements | MMLU-Pro, MATH-500, LiveCodeBench, FRAMES | Saturation (95%+) on original benchmarks makes compression-safety deltas unmeasurable |
| Statistical test | McNemar's (pairwise), bootstrap CIs | Standard for NLP accuracy comparisons; binary correct/wrong |

---

## Claims checklist (must be backed by real data before submission)

- [ ] "Safe compression rate averages X% across real agent conversations" → needs n=100 final
- [ ] "SCR varies by domain from X% to Y%"  → needs n=100 final + category labels
- [ ] "ContextPrune achieves Z% token reduction with <W% accuracy loss" → needs Exp 3
- [ ] "IAA between GPT-5.2 and Claude Sonnet 4.6: κ=X" → needs verification run
- [ ] "Compression preserves answers in X% of functional tests" → needs functional test
- [ ] "ContextPrune outperforms LLMLingua-2 on tool-heavy contexts" → needs Exp 3 on real data
- [ ] "Context-length stratified: X% savings at >2000 tokens" → needs Exp 3 stratified results

---

## Numbers to track (fill in as results come)

| Metric | Value | Source | Date |
|--------|-------|--------|------|
| Mean SCR (preliminary) | 60.6% ± 7.9% | n=55 manual annotations | 2026-02-24 |
| Median SCR | 68.0% | n=55 | 2026-02-24 |
| IAA κ | TBD | Claude Sonnet 4.6 re-annotation | pending |
| Answer preservation rate | TBD | Functional test | pending |
| MMLU-Pro accuracy (raw) | TBD | Exp 3 | pending |
| MMLU-Pro accuracy (ContextPrune) | TBD | Exp 3 | pending |
| Token reduction on MMLU-Pro | TBD | Exp 3 | pending |

---

## Writing order (recommended)

1. **Section 4** (CompressionSafety dataset) — write first, data nearly complete
2. **Section 3** (methodology) — well-defined, can write parallel to experiments
3. **Related Work** — can write now, independent of results
4. **Section 5** (experiments) — write after Exp 3 results
5. **Abstract + Introduction** — write last, when all claims are verified
6. **Conclusion** — write last

---

*Last updated: 2026-02-24 | n=55 annotations*
