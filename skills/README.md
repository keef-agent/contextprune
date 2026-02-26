# Agent Skills

This directory contains integration files for AI agent frameworks.

## `contextprune/` — OpenClaw / Codex / agentskills.io

A skill that lets AI agents (OpenClaw, Claude Code, Codex) use ContextPrune
directly from their session context.

**SKILL.md** — the skill definition (loaded by the agent framework)  
**references/science.md** — the academic foundations behind the approach  
**references/providers.md** — provider-specific setup notes

### What is an agent skill?

Agent skills are structured prompts + reference docs that teach an AI assistant
how to use a tool. They're the equivalent of a README, but written for an agent
rather than a human developer. The format is compatible with
[agentskills.io](https://agentskills.io) and any framework that supports skill
loading (OpenClaw, Codex, custom agents).

### Using the skill

```bash
# OpenClaw: copy to your workspace skills directory
cp -r skills/contextprune ~/.openclaw/workspace/skills/

# Codex: add to your agent's skill path
# Claude Code: use the /contextprune slash command in examples/claude-code-command.md
```
