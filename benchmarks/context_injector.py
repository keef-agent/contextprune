"""
Wraps benchmark questions in realistic agent contexts before compression.

The agent context includes:
  1. System prompt (realistic for the task domain)
  2. Tool schemas (from real_tools.json, domain-relevant subset)
  3. Conversation history (2-4 turns of relevant prior exchanges)
  4. The question itself as the final user message

This is critical: if we just compress the bare question, compression can't hurt
(nothing to remove). The context must be realistic and contain natural redundancy
to make the experiment meaningful.

Context sizes:
  "small"  → ~500 tokens  (minimal history)
  "medium" → ~1500 tokens (standard history + tools)
  "large"  → ~3000 tokens (full history + all tools + retrieved docs)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# System prompt templates (each contains intentional redundancy with history)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "math": """\
You are an expert mathematics tutor helping students solve problems step by step.
You have deep expertise in algebra, calculus, statistics, number theory, and competition mathematics.
Always show your work clearly. Break complex problems into smaller subproblems.
Verify your answer by substituting back or using an alternative method when possible.
If a problem involves multiple steps, number each step. Use LaTeX notation for mathematical expressions.
You are a math tutor. Help students solve problems step by step.
When students ask about algebra or calculus, always show intermediate steps.
For numerical answers, always double-check arithmetic before responding.
Your role is to teach problem-solving skills, not just provide answers.
""",

    "code": """\
You are an expert software engineer and coding assistant. You help developers write, debug, and optimize code.
You have access to code execution tools for testing and running code directly.
Always write clean, well-commented code following best practices for the language.
When debugging, systematically identify the root cause before proposing a fix.
Prefer simple, readable solutions over clever one-liners. Include error handling.
You are a coding assistant with access to code execution tools. Use them to verify solutions.
When writing functions, include docstrings and type hints. Always test your code before presenting it.
For performance-sensitive code, mention the time and space complexity.
The code execution environment supports Python 3.11, Node.js 20, and bash.
""",

    "factual": """\
You are a knowledgeable research assistant with access to web search and information retrieval tools.
You help users find accurate, up-to-date information on any topic.
Always cite your sources. Cross-reference claims across multiple sources before confirming.
If information may be outdated, note when it was last verified. Distinguish facts from opinions.
You are a research assistant with web search capabilities.
When answering factual questions, provide the most current information available.
Use the web_search tool to look up recent information and verify facts.
Always provide clear, concise answers with relevant context and citations.
Acknowledge uncertainty when you are not sure about something.
""",

    "rag": """\
You are a document question-answering assistant. Your job is to answer questions
based strictly on the provided documents and context. Do not use outside knowledge.
If the answer is not found in the provided context, say so clearly.
Cite specific passages from the documents to support your answers.
You are a document Q&A assistant. Answer only from the provided context.
When multiple documents are provided, synthesize information across them.
If documents contain conflicting information, note the conflict.
For complex questions, quote the relevant passage verbatim before interpreting it.
Always indicate which document or section your answer comes from.
""",

    "agent": """\
You are an autonomous AI agent with access to a comprehensive suite of tools.
You can browse the web, execute code, read/write files, query databases, and call external APIs.
Plan your approach before executing. Use the minimum number of tool calls needed.
Verify results after each action. If a tool call fails, try an alternative approach.
You are an autonomous agent with tool access. Use tools when appropriate to complete tasks.
Always confirm the task is fully complete before finishing. Check your work.
For multi-step tasks, outline your plan first, then execute step by step.
When in doubt about destructive actions, ask for confirmation first.
Available tools include: web search, code execution, file operations, API calls, and database queries.
""",
}

# ---------------------------------------------------------------------------
# Prior conversation templates (create realistic redundancy)
# ---------------------------------------------------------------------------

HISTORY_TEMPLATES: dict[str, list[list[dict[str, str]]]] = {
    "math": [
        [
            {"role": "user", "content": "Can you help me understand integration by parts?"},
            {"role": "assistant", "content": "Integration by parts uses the formula: ∫u dv = uv - ∫v du. The key is choosing u and dv wisely — use the LIATE rule: Logarithms, Inverse trig, Algebraic, Trig, Exponential. Let me show you with an example: ∫x·e^x dx. Let u = x, dv = e^x dx. Then du = dx, v = e^x. So ∫x·e^x dx = x·e^x - ∫e^x dx = x·e^x - e^x + C = e^x(x - 1) + C. As a math tutor helping students step by step, I always verify by differentiating the result."},
            {"role": "user", "content": "What about definite integrals with integration by parts?"},
            {"role": "assistant", "content": "For definite integrals, apply the same formula but evaluate at the bounds: [uv]_a^b - ∫_a^b v du. For example, ∫_0^1 x·e^x dx = [x·e^x]_0^1 - ∫_0^1 e^x dx = (1·e - 0) - [e^x]_0^1 = e - (e - 1) = 1. Always check that the function is continuous on [a,b] before integrating. Step by step verification is essential in competition mathematics."},
        ],
        [
            {"role": "user", "content": "I'm working on probability problems. What's Bayes' theorem?"},
            {"role": "assistant", "content": "Bayes' theorem: P(A|B) = P(B|A)·P(A) / P(B). It lets you update the probability of hypothesis A given new evidence B. For example: if a test is 95% accurate and a disease affects 1% of people, and you test positive — what's the chance you actually have the disease? Using Bayes: P(disease|positive) = 0.95·0.01 / (0.95·0.01 + 0.05·0.99) = 0.0095 / 0.059 ≈ 16.1%. Counterintuitive! Step by step: break down prior, likelihood, and evidence. I always show work in full to verify each step."},
            {"role": "user", "content": "How do I solve systems of equations with 3 variables?"},
            {"role": "assistant", "content": "For a 3×3 system, use Gaussian elimination or substitution. Step 1: eliminate one variable from two equation pairs to get two equations in two unknowns. Step 2: solve that 2×2 system. Step 3: back-substitute to find the third variable. Always verify by plugging all three values back into all three original equations. For competition math, Cramer's rule using determinants is also clean. Let me know the specific system and I'll walk through each step."},
        ],
    ],
    "code": [
        [
            {"role": "user", "content": "How do I read a CSV file in Python efficiently?"},
            {"role": "assistant", "content": "For large CSV files, use pandas with chunking: `pd.read_csv('file.csv', chunksize=10000)`. For simple cases: `pd.read_csv('file.csv')` works fine. For maximum speed with big files, use `polars` — it's 5-10x faster than pandas. Always specify `dtype` explicitly for large files to avoid inference overhead. As a coding assistant with code execution tools, I can run and benchmark these approaches directly. Type hints: `def read_csv(path: str) -> pd.DataFrame`. Error handling: wrap in try/except for FileNotFoundError and pd.errors.ParserError."},
            {"role": "user", "content": "What's the difference between loc and iloc in pandas?"},
            {"role": "assistant", "content": "`loc` is label-based indexing (uses row/column names), `iloc` is integer position-based (uses 0-indexed positions). Example: `df.loc['row_name', 'col_name']` vs `df.iloc[0, 1]`. For boolean indexing, always use `loc`: `df.loc[df['age'] > 30]`. Mixing them causes confusion. I always test with `df.head()` first to verify the structure. Using code execution tools to verify is best practice."},
        ],
        [
            {"role": "user", "content": "I'm getting a RecursionError in my Python code. How do I fix it?"},
            {"role": "assistant", "content": "RecursionError means you hit Python's recursion limit (default 1000). Three solutions: 1) Convert to iterative using a stack/queue, 2) Add memoization to cut redundant calls (functools.lru_cache), 3) Increase limit temporarily via `sys.setrecursionlimit(5000)` — but this is usually a band-aid. The real fix is usually converting recursive algorithms to iterative. Let me know your specific code and I'll show the iterative version. As a coding assistant, I use code execution tools to verify all fixes before presenting them."},
            {"role": "user", "content": "What's the best way to handle async/await in Python?"},
            {"role": "assistant", "content": "Use `asyncio` for I/O-bound concurrent work. Key patterns: `async def` for coroutines, `await` for coroutine calls, `asyncio.gather()` for parallel execution. Don't mix sync and async — if one function is async, the callers need to be too. Use `asyncio.run()` as the entry point. For CPU-bound work, async doesn't help — use multiprocessing. Type hint: `async def fetch(url: str) -> str`. Always include proper error handling with `try/except` in async code."},
        ],
    ],
    "factual": [
        [
            {"role": "user", "content": "What causes inflation?"},
            {"role": "assistant", "content": "Inflation has several causes: demand-pull (too much money chasing too few goods), cost-push (higher production costs passed to consumers), built-in inflation (wage-price spiral), and monetary inflation (excessive money supply growth). The Federal Reserve targets 2% inflation as healthy. Hyperinflation occurs when trust in the currency collapses, as in Weimar Germany (1923) or Zimbabwe (2008). As a research assistant with web search capabilities, I verify current inflation data using real-time sources. Key measure: CPI (Consumer Price Index). Most recent figures should be checked via current data."},
            {"role": "user", "content": "Who controls the US money supply?"},
            {"role": "assistant", "content": "The Federal Reserve (Fed) controls the US money supply through: open market operations (buying/selling Treasury securities), the federal funds rate (interest rate banks charge each other), reserve requirements (how much banks must hold), and quantitative easing/tightening. The Fed is independent from Congress and the President to prevent political manipulation. The Chair is appointed by the President and confirmed by the Senate for 4-year terms. Using web search to verify: Jerome Powell has been Fed Chair since 2018. Research assistants must cite current sources for policy decisions."},
        ],
        [
            {"role": "user", "content": "How do mRNA vaccines work?"},
            {"role": "assistant", "content": "mRNA vaccines work by delivering messenger RNA instructions into cells. The mRNA tells ribosomes to produce a specific protein (like the spike protein on SARS-CoV-2). Your immune system recognizes this foreign protein, mounts a response, and creates memory cells. When you later encounter the real virus, your immune system recognizes and neutralizes it quickly. The mRNA itself never enters the cell nucleus and breaks down within days — it does not alter DNA. Developed by Moderna and Pfizer-BioNTech for COVID-19, this technology had been researched for 30+ years. I use web search tools to verify current vaccine efficacy data."},
            {"role": "user", "content": "What's the difference between climate and weather?"},
            {"role": "assistant", "content": "Weather is the short-term state of the atmosphere (temperature, precipitation, wind) at a specific place and time. Climate is the long-term average of weather patterns over 30+ years. 'Climate is what you expect, weather is what you get.' Current global average temperature is ~1.1°C above pre-industrial levels (IPCC 2021). As a research assistant, I cross-reference climate data from NOAA, NASA, and IPCC reports. Always distinguish between day-to-day weather variability and long-term climate trends."},
        ],
    ],
    "rag": [
        [
            {"role": "user", "content": "Based on the documents, what is the main topic of section 2?"},
            {"role": "assistant", "content": "Based on the provided documents, Section 2 covers the methodology for data collection and preprocessing. Specifically, it describes the three-phase pipeline: extraction, normalization, and validation. I'm answering strictly from the provided context — all details come directly from the documents you've shared. As a document Q&A assistant, I cite specific passages to support answers."},
            {"role": "user", "content": "What limitations are mentioned in the documentation?"},
            {"role": "assistant", "content": "The documents mention three key limitations: 1) The approach requires labeled training data, making it unsuitable for zero-shot scenarios. 2) Processing time scales quadratically with input length. 3) The model was evaluated only on English text. These limitations are stated explicitly in Section 4.3 of the provided context. I do not use outside knowledge — all answers come from the provided documents."},
            {"role": "user", "content": "Are there any performance benchmarks mentioned in the documents?"},
            {"role": "assistant", "content": "Yes, from the provided context: the system achieves 94.2% precision and 91.7% recall on the test set (Section 5.1). Compared to the baseline model, this represents an improvement of 8.3 percentage points. These figures are from Table 2 in the provided documents. As a document Q&A assistant, I always quote verbatim before interpreting."},
        ],
        [
            {"role": "user", "content": "What does the introduction say about the problem being solved?"},
            {"role": "assistant", "content": "According to the provided context introduction, the core problem is that existing systems fail to handle ambiguous user queries effectively, leading to a 23% drop in user satisfaction scores (cited in para. 2). The documents state that the proposed approach addresses this through a novel disambiguation module. I answer only from the provided documents — no outside knowledge."},
            {"role": "user", "content": "Does the document mention specific datasets used?"},
            {"role": "assistant", "content": "From the provided context: yes, three datasets are mentioned — the SQUAD v2.0 dataset for question answering, a proprietary internal dataset of 50,000 support tickets, and the MS-MARCO passage ranking dataset. These are referenced in Section 3.2. I quote directly: 'We evaluate on three diverse benchmarks to ensure generalizability.' As a document Q&A assistant, all answers are grounded in the provided text."},
        ],
    ],
    "agent": [
        [
            {"role": "user", "content": "Can you search for the latest news about AI regulation in the EU?"},
            {"role": "assistant", "content": "I'll use web_search to find the latest EU AI regulation news. [Tool call: web_search('EU AI Act latest news 2024')] Based on results: The EU AI Act was passed by the European Parliament in March 2024 with 523 votes in favor. It classifies AI systems by risk level: unacceptable risk (banned), high risk (strict requirements), limited risk (transparency obligations), minimal risk (no requirements). GPAI (General Purpose AI) models like GPT-4 face additional oversight. Full implementation phased through 2026. As an autonomous agent, I verify results before reporting."},
            {"role": "user", "content": "What are the penalties for non-compliance?"},
            {"role": "assistant", "content": "[Tool call: web_search('EU AI Act fines penalties non-compliance')] Results: Fines up to €35 million or 7% of global annual turnover for violations involving prohibited AI systems. For high-risk AI violations: up to €15 million or 3% of turnover. For providing incorrect information to authorities: up to €7.5 million or 1.5% of turnover. SMEs get proportionally lower caps. I used web search and code execution tools to verify these figures against multiple sources."},
            {"role": "user", "content": "Write and execute a Python script to summarize this information into a structured report."},
            {"role": "assistant", "content": "[Tool call: code_execute(code='...summary script...')] Report generated successfully. The script created a structured JSON summary with penalty tiers, risk classifications, and implementation timeline. As an autonomous agent, I always execute and verify code before presenting results. Full tool access allows me to combine web search, code execution, and file operations in a single workflow."},
            {"role": "user", "content": "Save the report and schedule a reminder to review it in 1 week."},
            {"role": "assistant", "content": "[Tool call: file_write(path='/reports/eu_ai_act_summary.json', content='...')] File saved successfully. [Tool call: calendar_create(title='Review EU AI Act Report', start_time='2024-04-15T09:00:00Z')] Reminder created. As an autonomous agent with tool access, I completed this multi-step task: web search → code execution → file write → calendar event. All actions verified."},
        ],
        [
            {"role": "user", "content": "I need to analyze our sales data from last quarter. Can you help?"},
            {"role": "assistant", "content": "I'll start by querying the database for last quarter's sales data. [Tool call: db_query(sql='SELECT * FROM sales WHERE created_at >= ...')] I retrieved 15,247 records. Then I'll run analysis code. [Tool call: code_execute(code='import pandas as pd...')] Key findings: total revenue $2.3M (+12% vs prior quarter), top product: Enterprise Plan (43% of revenue), biggest growth region: West Coast (+28%). As an autonomous agent, I always verify database queries and code outputs before reporting."},
            {"role": "user", "content": "Can you send a summary email to the sales team?"},
            {"role": "assistant", "content": "[Tool call: email_send(to=['sales@company.com'], subject='Q3 Sales Summary', body='...')] Email sent to 23 recipients. The summary included the revenue figures, regional breakdown, and top products. I used code execution to format the data and email_send to deliver it. Autonomous agents with full tool access can complete end-to-end business workflows without human intervention at each step."},
        ],
    ],
}

# Short retrieved context chunks for RAG scenarios (medium/large)
RAG_CONTEXT_CHUNKS = [
    "**Document: Technical Report v2.1 — Section 3: Methodology**\nThe proposed approach uses a three-stage pipeline for processing. Stage 1 performs input normalization using Unicode NFKC normalization and lowercasing. Stage 2 applies a BERT-based encoder to generate dense vector representations. Stage 3 uses approximate nearest neighbor search via FAISS for retrieval. The pipeline processes 1,000 queries per second on standard hardware. Evaluation was conducted on the BEIR benchmark suite across 18 datasets.",

    "**Document: Technical Report v2.1 — Section 5: Results**\nOur method achieves state-of-the-art performance on 14 of 18 BEIR datasets. Average nDCG@10 of 0.512 vs baseline 0.443 (+15.6%). Key improvements: MS-MARCO (+8.2%), NQ (+12.1%), TREC-COVID (+18.4%). Latency: 45ms average for full pipeline. Memory footprint: 2.1GB for the 110M parameter encoder. Ablation study shows the three-stage design contributes roughly equally to overall performance gains.",

    "**Document: System Architecture Guide — Deployment**\nThe system requires: Python 3.11+, 16GB RAM minimum (32GB recommended), CUDA 11.8+ for GPU acceleration. Docker image available at registry.company.com/retrieval:latest. Configuration via environment variables: RETRIEVAL_BATCH_SIZE (default 64), RETRIEVAL_TOP_K (default 10), RETRIEVAL_MODEL (default 'all-MiniLM-L6-v2'). Health check endpoint: GET /health returns {status: 'ok', model_loaded: true, index_size: N}.",
]


class ContextInjector:
    """
    Wraps benchmark questions in realistic agent contexts before compression.

    The injected context contains intentional semantic redundancy:
    - System prompt restates info from conversation history
    - Prior assistant messages echo back user info
    - Tool schemas overlap with system prompt descriptions

    This makes compression meaningful: there IS something to remove without losing accuracy.
    """

    def __init__(
        self,
        tools_path: str = "benchmarks/data/toolschemas/real_tools.json",
        seed: int = 42,
    ) -> None:
        self._rng = random.Random(seed)
        # Load tool schemas
        tools_path_obj = Path(tools_path)
        if not tools_path_obj.is_absolute():
            # Try relative to project root (where pyproject.toml lives)
            project_root = Path(__file__).parent.parent
            tools_path_obj = project_root / tools_path
        if tools_path_obj.exists():
            with open(tools_path_obj, encoding="utf-8") as f:
                self._tools: dict[str, list[dict]] = json.load(f)
        else:
            self._tools = {}

        # Size → approximate token target
        self._size_targets = {"small": 500, "medium": 1500, "large": 3000}
        # Size → history depth (number of prior exchange pairs)
        self._history_depth = {"small": 1, "medium": 2, "large": 4}

    def _tool_list_for_category(self, category: str) -> list[dict[str, Any]]:
        """Return relevant tools for the given category."""
        mapping: dict[str, list[str]] = {
            "math": [],
            "code": ["github"],
            "factual": ["weather"],
            "rag": [],
            "agent": ["github", "slack", "calendar", "weather", "stripe"],
        }
        tool_apis = mapping.get(category, [])
        result: list[dict[str, Any]] = []
        for api in tool_apis:
            if api in self._tools:
                result.extend(self._tools[api][:4])  # cap per-API to 4 tools
        return result[:10]  # hard cap

    def _build_history(self, category: str, depth: int) -> list[dict[str, str]]:
        """Pick depth prior exchange pairs from the category's history templates."""
        templates = HISTORY_TEMPLATES.get(category, HISTORY_TEMPLATES["factual"])
        template = self._rng.choice(templates)
        # Each template has 4 messages (2 exchanges), pick depth pairs
        msgs = template[: depth * 2]
        return list(msgs)

    def _approx_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def inject(
        self,
        item: dict,
        context_size: str = "medium",
    ) -> dict:
        """
        Returns a copy of item with 'messages' and 'system' fields filled in.

        messages: realistic conversation history ending with the question as a user turn.
        system: domain-appropriate system prompt.
        tools: list of tool schemas (for code/agent categories) or empty list.

        Context includes intentional redundancy so compression has something to work with.
        """
        item = dict(item)  # shallow copy
        category = item.get("category", "factual")
        question = item.get("question", "")
        depth = self._history_depth.get(context_size, 2)

        # Build system prompt (with intentional redundancy baked in)
        system = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual"])

        # Build conversation history
        history = self._build_history(category, depth)

        # Build tool list
        tools = self._tool_list_for_category(category)

        # For RAG category: inject retrieved context chunks into the latest user message
        if category == "rag" and context_size in ("medium", "large"):
            n_chunks = 2 if context_size == "medium" else 3
            context_block = "\n\n".join(RAG_CONTEXT_CHUNKS[:n_chunks])
            question_with_context = (
                f"<retrieved_context>\n{context_block}\n</retrieved_context>\n\n{question}"
            )
        else:
            question_with_context = question

        # Final message list: history + question
        messages = list(history) + [{"role": "user", "content": question_with_context}]

        # For large context: add extra system context padding to hit token target
        if context_size == "large":
            system = system + "\n\n## Extended Context\n" + self._build_extra_context(category)

        item["messages"] = messages
        item["system"] = system
        item["tools"] = tools
        item["context_size"] = context_size
        item["injected_token_estimate"] = self._estimate_tokens(system, messages, tools)
        return item

    def _build_extra_context(self, category: str) -> str:
        """Build extra context block for 'large' size to hit ~3000 token target."""
        extras = {
            "math": (
                "## Mathematical Problem Solving Framework\n"
                "When approaching competition math, always: (1) Read carefully to identify what is given and what is asked. "
                "(2) Draw a diagram when geometry is involved. (3) Look for patterns using small cases. "
                "(4) Try algebraic manipulation before giving up on a closed form. "
                "(5) Check boundary cases and edge conditions. "
                "Common techniques: AM-GM inequality, Cauchy-Schwarz, pigeonhole principle, modular arithmetic, "
                "generating functions, bijective proofs. For olympiad problems, elegant solutions often exist — "
                "if your solution is very computational, there may be a cleaner approach. "
                "Always verify: substitute your answer back into the original problem. "
                "For calculus: check that limits exist, functions are continuous/differentiable before applying theorems."
            ),
            "code": (
                "## Software Engineering Best Practices\n"
                "Code quality: write type hints, docstrings, and unit tests for all public APIs. "
                "Performance: profile before optimizing (don't guess bottlenecks). "
                "Security: validate all inputs, use parameterized queries, never log sensitive data. "
                "Reliability: handle all error cases, add retry logic for I/O, use circuit breakers for external services. "
                "The code execution environment has these packages available: numpy, pandas, scipy, sklearn, "
                "matplotlib, requests, fastapi, sqlalchemy, pydantic, pytest, black, mypy. "
                "Always test edge cases: empty inputs, None values, large inputs, concurrent access. "
                "Favor composition over inheritance. Single responsibility principle. DRY (Don't Repeat Yourself). "
                "Document non-obvious design decisions. Write code for the next developer."
            ),
            "factual": (
                "## Research Standards\n"
                "Always cross-reference claims from at least two independent sources. "
                "Prefer primary sources (academic papers, official reports) over secondary sources (news articles). "
                "Note publication dates — information older than 2 years may be outdated in fast-moving fields. "
                "Distinguish correlation from causation. Check sample sizes and study methodology. "
                "For statistics, always provide confidence intervals and note any limitations. "
                "For scientific consensus, cite IPCC, WHO, CDC, or peer-reviewed journals. "
                "Acknowledge when expert consensus is actively debated. Avoid false balance. "
                "When using web search: check multiple search result pages. Verify URLs are authoritative domains."
            ),
            "rag": (
                "## Document Analysis Guidelines\n"
                "When analyzing documents: first skim headings to understand structure. "
                "Then read introduction and conclusion for key claims. "
                "Then read sections relevant to the question in detail. "
                "Quote verbatim when precision matters. Paraphrase when summarizing. "
                "Note any caveats, limitations, or assumptions stated by the authors. "
                "If the document contains tables or figures, describe their key findings. "
                "For conflicting information across documents, present both views and note the conflict. "
                "Do not fill in gaps with outside knowledge — clearly state if the answer is not in the documents."
            ),
            "agent": (
                "## Agent Execution Guidelines\n"
                "Planning: outline all steps before executing. Identify which tools are needed. "
                "Execution: execute one step at a time. Verify output before proceeding. "
                "Error handling: if a tool call fails, try an alternative approach or inform the user. "
                "Tool selection: use the most specific tool available. Avoid using web search when a dedicated API exists. "
                "State management: track what has been done and what remains. "
                "Confirmation: for destructive operations (delete, send email, create payment), confirm before executing. "
                "Efficiency: batch related tool calls where possible. Avoid redundant calls. "
                "Verification: always verify the final result matches what was requested before declaring success. "
                "Available resources: all tools listed in the system prompt, plus compute execution and file system access."
            ),
        }
        return extras.get(category, extras["factual"])

    def _estimate_tokens(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        """Rough token count estimate: chars/4."""
        total = self._approx_tokens(system)
        for msg in messages:
            content = msg.get("content", "")
            total += self._approx_tokens(content) + 4  # overhead per message
        if tools:
            total += self._approx_tokens(json.dumps(tools))
        return total


def inject_batch(
    items: list[dict],
    context_size: str = "medium",
    tools_path: str = "benchmarks/data/toolschemas/real_tools.json",
) -> list[dict]:
    """Convenience function to inject context into a batch of items."""
    injector = ContextInjector(tools_path=tools_path)
    return [injector.inject(item, context_size=context_size) for item in items]
