# Integrations

Framework-specific setup files for ContextPrune.

| Framework | Directory | Format |
|---|---|---|
| Claude Code | `claude-code/` | Slash command (`~/.claude/commands/`) |
| OpenClaw | `openclaw/` | Agent skill (SKILL.md format) |
| Codex | `codex/` | Agent skill (SKILL.md format) |

## Claude Code

Copy the slash command to your Claude commands directory:

```bash
cp integrations/claude-code/contextprune.md ~/.claude/commands/contextprune.md
```

Then type `/contextprune` in Claude Code to check proxy status and view stats.

## OpenClaw

Copy the skill to your OpenClaw workspace:

```bash
cp -r integrations/openclaw ~/.openclaw/workspace/skills/contextprune
```

OpenClaw will pick it up automatically on next session start.

## Codex

Copy the skill to your Codex workspace:

```bash
cp -r integrations/codex ~/.codex/workspace/skills/contextprune
```

Or for any [agentskills.io](https://agentskills.io)-compatible framework:

```bash
cp -r integrations/openclaw ~/.agents/skills/contextprune
```

## Shared References

- `openclaw/references/science.md` — academic foundations, real-world results
- `openclaw/references/providers.md` — provider-specific routing (Grok, Gemini, OpenRouter, etc.)
- `oauth-subscriptions.md` — using ContextPrune with claude.ai / Codex OAuth subscriptions
