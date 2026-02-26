# Changelog

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
