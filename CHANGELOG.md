# Changelog

## [0.1.1] — 2026-02-26

### Added
- `SemanticDeduplicator(protect_system=True)` — system prompt is now returned unchanged by default. All system chunks are still added to the seen pool so message-level duplicates are stripped, but the system prompt itself is never modified. Motivated by "The Pitfalls of KV Cache Compression" (arXiv 2510.00231, 2025), which showed compression can silently cause LLMs to ignore instructions.
- `_is_instructional()` heuristic — detects imperative instruction patterns ("Never", "Do not", "Always", "You must", etc.) for use in future guardrail extensions.
- 17 new tests in `tests/test_protect_system.py` covering the protect_system flag, the `_is_instructional` helper, and within-system dedup behavior.

### Changed
- `SemanticDeduplicator` with `protect_system=False`: system chunks are now processed one at a time (instead of batch) so within-system duplicates are properly detected if opting out of protection.

## [0.1.0] — 2026-02-26

### Added
- Semantic deduplication proxy via `POST /v1/messages` (Anthropic Messages API)
- `POST /v1/chat/completions` endpoint (OpenAI, Grok, OpenRouter, Google Gemini compat)
- `POST /v1/responses` endpoint (OpenAI Responses API, Codex)
- `GET /` health check
- `contextprune serve` CLI command with `--port`, `--threshold`, `--log`, `--openai-target` flags
- Stats logging to `~/.contextprune/stats.jsonl`
- Dry-run mode via `__contextprune_dry_run` flag
- Agent skill for OpenClaw, Codex, and agentskills.io-compatible frameworks
- Claude Code slash command (`/contextprune`)
