> **Copy this to `~/.claude/commands/contextprune.md` to use as a Claude Code slash command.**
> Then type `/contextprune` in Claude Code to check proxy status and view compression stats.

---

# ContextPrune Status

Check the status of the ContextPrune proxy and show recent compression statistics.

## Steps

**1. Check if the proxy is running:**

Run this command:
```
ps aux | grep "contextprune.proxy" | grep -v grep
```

**2. If the proxy is NOT running, start it:**

If the output is empty, start the proxy. Try pip-installed version first:
```
contextprune serve --port 8899 &
```

If that fails (command not found), start from the repo:
```
cd /path/to/contextprune && .venv/bin/python3 -m contextprune.proxy --port 8899 --log &
```

Tell the user: "ContextPrune proxy started on port 8899."

**3. Show recent compression stats:**

Run this command and display the output:
```
tail -20 ~/.contextprune/stats.jsonl 2>/dev/null | python3 -c "import sys,json; rows=[json.loads(l) for l in sys.stdin]; total_orig=sum(r['original_tokens'] for r in rows); total_comp=sum(r['compressed_tokens'] for r in rows); saved=total_orig-total_comp; ratio=total_comp/total_orig if total_orig else 1; print(f'Last {len(rows)} requests: {saved:,} tokens saved ({(1-ratio)*100:.1f}% reduction)')"
```

If no stats file exists or it's empty, say: "No compression history yet."

**4. Check the ANTHROPIC_BASE_URL environment variable:**

Run:
```
echo $ANTHROPIC_BASE_URL
```

If it is not set to `http://localhost:8899`, remind the user:

> ⚠️ To route Claude API calls through ContextPrune, run:
> ```
> export ANTHROPIC_BASE_URL=http://localhost:8899
> ```

**5. Print a one-line summary:**

Using the stats from step 3, calculate average compression and print:

> ContextPrune is active. X% average compression across Y requests.

(Replace X with the reduction percentage and Y with the number of requests. If no stats yet, say "0% compression across 0 requests.")
