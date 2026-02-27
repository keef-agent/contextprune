# Changelog

## [0.1.3] — 2026-02-26

### Fixed
- **Critical: Claude Code and all array-content messages were silently skipped.** The deduplicator only handled `isinstance(content, str)`. The Anthropic API (and every modern Claude Code session) sends message content as typed block arrays (`text`, `tool_use`, `tool_result`). Every Claude Code message hit the `else: pass through` branch — zero deduplication ever occurred. Fix: `dedup_block_list()` recursively processes text blocks and tool_result blocks while passing tool_use/image blocks through unchanged.

### Added
- 5 new tests in `tests/test_dedup.py::TestBlockListDedup` covering text blocks, tool_result string/list content, tool_use passthrough, and mixed content arrays.

## [0.1.2] — 2026-02-26

### Fixed
- `httpx.StreamClosed` crash on streaming responses (`stream=True`). The proxy was closing the httpx client and response stream immediately after returning `StreamingResponse(resp.aiter_bytes())` — before starlette consumed any chunks. Fix: context managers now live inside the generator via `_make_streaming_response()`, so the stream stays open until the last byte is yielded.
- `HPE_UNEXPECTED_CONTENT_LENGTH` error in clients (including Claude Code) caused by forwarding the upstream `content-length` header on chunked streaming responses. Stripped via `_STRIP_RESPONSE_HEADERS`.

### Added
- 3 regression tests in `tests/test_proxy_streaming.py` covering chunk consumption, header stripping, and upstream status code forwarding.

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
