# Contributing

PRs welcome. Keep it simple.

## Setup

```bash
git clone https://github.com/keef-agent/contextprune
cd contextprune
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest tests/ -x -q
```

## Guidelines

- New features need a test
- Keep the proxy fast — benchmark before/after if touching dedup or embedding code
- No external API calls in tests (mock them)
- Run `pytest` before submitting a PR
