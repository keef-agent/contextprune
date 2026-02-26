"""
ContextPrune Proxy Server

Drop-in Anthropic API proxy that deduplicates context before forwarding.

Usage:
    python -m contextprune.proxy --port 8899

Then point any Anthropic client at http://localhost:8899 instead of
https://api.anthropic.com and your requests are automatically compressed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .dedup import SemanticDeduplicator
from .tokenizer import count_message_tokens, count_system_tokens

# --------------------------------------------------------------------------- #
# Globals (set at startup via serve())                                         #
# --------------------------------------------------------------------------- #
_deduplicator: Optional[SemanticDeduplicator] = None
_enable_logging: bool = True
_stats_path: Path = Path.home() / ".contextprune" / "stats.jsonl"
_request_counter: int = 0

log = logging.getLogger("contextprune.proxy")

app = FastAPI(title="ContextPrune Proxy", version="0.1.0")

ANTHROPIC_BASE = "https://api.anthropic.com"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _approx_tokens(text: str) -> int:
    """~4 chars per token."""
    return max(1, len(text) // 4)


def _count_body_tokens(body: Dict[str, Any]) -> int:
    messages = body.get("messages", [])
    system = body.get("system", "")
    if isinstance(system, list):
        # Blocks format
        system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    msg_tok = count_message_tokens(messages)
    sys_tok = count_system_tokens(system if isinstance(system, str) else "")
    return msg_tok + sys_tok


def _write_stat(record: Dict[str, Any]) -> None:
    """Append a single stat record to the JSONL log."""
    if not _enable_logging:
        return
    try:
        _stats_path.parent.mkdir(parents=True, exist_ok=True)
        with _stats_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        log.warning("Failed to write stat: %s", exc)


def _extract_text_content(body: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Extract messages and system string from the request body."""
    messages = body.get("messages", [])
    system = body.get("system", None)
    if isinstance(system, list):
        # Anthropic can send system as a list of content blocks
        system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    return messages, system


def _run_dedup(
    messages: List[Dict[str, Any]],
    system: Optional[str],
) -> tuple[List[Dict[str, Any]], Optional[str], int, int, int, float]:
    """Run deduplication and return (new_msgs, new_sys, orig_tok, comp_tok, removed, ratio)."""
    assert _deduplicator is not None
    orig_tok = count_message_tokens(messages) + count_system_tokens(system or "")
    try:
        new_messages, new_system, removed = _deduplicator.deduplicate(messages, system=system)
    except Exception as exc:
        log.warning("Deduplication failed (%s) — passing original through", exc)
        return messages, system, orig_tok, orig_tok, 0, 1.0

    comp_tok = count_message_tokens(new_messages) + count_system_tokens(new_system or "")
    ratio = comp_tok / orig_tok if orig_tok > 0 else 1.0
    return new_messages, new_system, orig_tok, comp_tok, removed, ratio


# --------------------------------------------------------------------------- #
# Route: POST /v1/messages                                                     #
# --------------------------------------------------------------------------- #

@app.post("/v1/messages")
async def proxy_messages(request: Request) -> Response:
    global _request_counter
    _request_counter += 1
    q = _request_counter

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # ---- Dry-run mode (testing / integration) ----------------------------- #
    is_dry_run = body.pop("__contextprune_dry_run", False)

    model = body.get("model", "unknown")
    is_stream = body.get("stream", False)

    # ---- Skip dedup for streaming (pass through unchanged) ---------------- #
    if is_stream:
        if not is_dry_run:
            headers = dict(request.headers)
            headers.pop("host", None)
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = client.stream(
                    "POST",
                    f"{ANTHROPIC_BASE}/v1/messages",
                    headers=headers,
                    content=raw_body,
                )
                async with upstream as resp:
                    return StreamingResponse(
                        resp.aiter_bytes(),
                        status_code=resp.status_code,
                        headers=dict(resp.headers),
                        media_type=resp.headers.get("content-type", "text/event-stream"),
                    )

    # ---- Deduplication ---------------------------------------------------- #
    messages, system = _extract_text_content(body)
    new_messages, new_system, orig_tok, comp_tok, removed, ratio = _run_dedup(messages, system)
    saved = orig_tok - comp_tok

    # Stdout log
    print(
        f"[ContextPrune] Q{q} ratio={ratio:.2f} removed={removed}sents saved={saved}tok",
        flush=True,
    )

    # File log
    stat_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "original_tokens": orig_tok,
        "compressed_tokens": comp_tok,
        "ratio": round(ratio, 4),
        "sentences_removed": removed,
    }
    _write_stat(stat_record)

    # ---- Dry-run: return stats only --------------------------------------- #
    if is_dry_run:
        return JSONResponse({
            "contextprune": {
                "original_tokens": orig_tok,
                "compressed_tokens": comp_tok,
                "ratio": round(ratio, 4),
                "sentences_removed": removed,
            }
        })

    # ---- Build modified body ---------------------------------------------- #
    modified_body = dict(body)
    modified_body["messages"] = new_messages
    if new_system is not None:
        modified_body["system"] = new_system

    # ---- Forward to Anthropic -------------------------------------------- #
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)  # let httpx recalculate

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ANTHROPIC_BASE}/v1/messages",
                headers=headers,
                json=modified_body,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except Exception as exc:
        log.error("Upstream request failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=502)


# --------------------------------------------------------------------------- #
# Health check                                                                 #
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "contextprune-proxy"})


# --------------------------------------------------------------------------- #
# serve() — public API                                                         #
# --------------------------------------------------------------------------- #

def serve(
    port: int = 8899,
    threshold: float = 0.82,
    enable_log: bool = True,
    host: str = "127.0.0.1",
) -> None:
    """Start the ContextPrune proxy server.

    Args:
        port: Port to listen on. Default 8899.
        threshold: Semantic similarity threshold for deduplication. Default 0.82.
        enable_log: Whether to log stats to ~/.contextprune/stats.jsonl.
        host: Host to bind to. Default 127.0.0.1 (localhost only).
    """
    import uvicorn

    global _deduplicator, _enable_logging
    _deduplicator = SemanticDeduplicator(similarity_threshold=threshold)
    _enable_logging = enable_log

    print(f"[ContextPrune] Proxy starting on http://{host}:{port}", flush=True)
    print(f"[ContextPrune] Threshold={threshold} | Logging={'ON' if enable_log else 'OFF'}", flush=True)
    if enable_log:
        print(f"[ContextPrune] Stats → {_stats_path}", flush=True)

    uvicorn.run(app, host=host, port=port, log_level="warning")


# --------------------------------------------------------------------------- #
# __main__ entry point                                                         #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m contextprune.proxy",
        description="ContextPrune Proxy — drop-in Anthropic API proxy with deduplication",
    )
    parser.add_argument(
        "--port", type=int, default=8899, help="Port to listen on (default: 8899)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Similarity threshold for deduplication (default: 0.82)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable stats file logging",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    serve(
        port=args.port,
        threshold=args.threshold,
        enable_log=not args.no_log,
        host=args.host,
    )
