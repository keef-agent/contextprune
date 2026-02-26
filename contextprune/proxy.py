"""
ContextPrune Proxy Server

Drop-in Anthropic API proxy that deduplicates context before forwarding.
Also supports OpenAI-compatible /v1/chat/completions endpoint.

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
_openai_target: str = os.environ.get("CONTEXTPRUNE_OPENAI_TARGET", "https://api.openai.com")

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
# Route: GET /                                                                 #
# --------------------------------------------------------------------------- #

@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "version": "0.1.0",
        "endpoints": ["/v1/messages", "/v1/chat/completions", "/v1/responses"],
    })


# --------------------------------------------------------------------------- #
# Route: POST /v1/messages  (Anthropic)                                        #
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

    # ---- Deduplication (runs on input for ALL requests, including streaming) #
    messages, system = _extract_text_content(body)
    new_messages, new_system, orig_tok, comp_tok, removed, ratio = _run_dedup(messages, system)
    saved = orig_tok - comp_tok

    # Stdout log
    if is_stream:
        print(
            "[ContextPrune] Streaming request — dedup applied to input, streaming response passed through",
            flush=True,
        )
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
        if is_stream:
            # Dedup was applied to input; forward deduplicated body as streaming request
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = client.stream(
                    "POST",
                    f"{ANTHROPIC_BASE}/v1/messages",
                    headers=headers,
                    json=modified_body,
                )
                async with upstream as resp:
                    return StreamingResponse(
                        resp.aiter_bytes(),
                        status_code=resp.status_code,
                        headers=dict(resp.headers),
                        media_type=resp.headers.get("content-type", "text/event-stream"),
                    )
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
# Route: POST /v1/chat/completions  (OpenAI-compatible)                        #
# --------------------------------------------------------------------------- #

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    global _request_counter
    _request_counter += 1
    q = _request_counter

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # ---- Dry-run mode ----------------------------------------------------- #
    is_dry_run = body.pop("__contextprune_dry_run", False)

    model = body.get("model", "unknown")
    is_stream = body.get("stream", False)

    # ---- Determine target URL --------------------------------------------- #
    target_base = request.headers.get("x-target-url", _openai_target).rstrip("/")

    # ---- Extract messages — OpenAI puts system inside the messages array -- #
    messages: List[Dict[str, Any]] = body.get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system_messages = [m for m in messages if m.get("role") != "system"]

    # Combine system messages into a single system string
    system: Optional[str] = None
    if system_messages:
        system = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str)
            else " ".join(b.get("text", "") for b in m.get("content", []) if isinstance(b, dict))
            for m in system_messages
        ).strip() or None

    # ---- Deduplication (runs on input for ALL requests, including streaming) #
    new_messages, new_system, orig_tok, comp_tok, removed, ratio = _run_dedup(
        non_system_messages, system
    )
    saved = orig_tok - comp_tok

    # Reconstruct OpenAI-format messages array: system first, then deduplicated messages
    final_messages: List[Dict[str, Any]] = []
    if new_system:
        final_messages.append({"role": "system", "content": new_system})
    final_messages.extend(new_messages)

    # Stdout log
    if is_stream:
        print(
            "[ContextPrune] Streaming request — dedup applied to input, streaming response passed through",
            flush=True,
        )
    print(
        f"[ContextPrune/OAI] Q{q} ratio={ratio:.2f} removed={removed}sents saved={saved}tok",
        flush=True,
    )

    # File log
    stat_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "endpoint": "chat/completions",
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
    modified_body["messages"] = final_messages

    # ---- Forward to OpenAI-compatible target ------------------------------ #
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("x-target-url", None)  # don't forward our custom header

    try:
        if is_stream:
            # Dedup was applied to input; forward deduplicated body as streaming request
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = client.stream(
                    "POST",
                    f"{target_base}/v1/chat/completions",
                    headers=headers,
                    json=modified_body,
                )
                async with upstream as resp:
                    return StreamingResponse(
                        resp.aiter_bytes(),
                        status_code=resp.status_code,
                        headers=dict(resp.headers),
                        media_type=resp.headers.get("content-type", "text/event-stream"),
                    )
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{target_base}/v1/chat/completions",
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
# Route: POST /v1/responses  (OpenAI Responses API / Codex OAuth)              #
# --------------------------------------------------------------------------- #

@app.post("/v1/responses")
async def proxy_responses(request: Request) -> Response:
    """Proxy for the OpenAI Responses API (used by Agents SDK, Codex CLI, OpenClaw openai-codex).

    The Responses API differs from chat/completions:
      - Request uses ``input`` array instead of ``messages``
      - System prompt is a top-level ``instructions`` field
      - Response has ``output`` array instead of ``choices``

    ContextPrune deduplicates the ``input`` items (same role/content structure as messages)
    and updates ``instructions`` with the deduplicated system text before forwarding.
    """
    global _request_counter
    _request_counter += 1
    q = _request_counter

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # ---- Dry-run mode ----------------------------------------------------- #
    is_dry_run = body.pop("__contextprune_dry_run", False)

    model = body.get("model", "unknown")
    is_stream = body.get("stream", False)

    # ---- Determine target URL --------------------------------------------- #
    target_base = _openai_target.rstrip("/")

    # ---- Extract input and instructions ------------------------------------ #
    # ``input`` items use the same {role, content} structure as messages.
    # Input items may also be plain strings; wrap those as user messages for dedup.
    raw_input: List[Any] = body.get("input", [])
    messages: List[Dict[str, Any]] = []
    for item in raw_input:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
        elif isinstance(item, dict):
            messages.append(item)
        # else: skip unexpected types

    # ``instructions`` is the system prompt field in the Responses API
    system: Optional[str] = body.get("instructions") or None
    if isinstance(system, str) and not system.strip():
        system = None

    # ---- Deduplication ---------------------------------------------------- #
    new_messages, new_system, orig_tok, comp_tok, removed, ratio = _run_dedup(messages, system)
    saved = orig_tok - comp_tok

    # Reconstruct input array preserving original string items where possible
    new_input: List[Any] = []
    for i, item in enumerate(raw_input):
        if i < len(new_messages):
            # If original was a plain string and content hasn't changed, keep as string
            if isinstance(item, str) and new_messages[i].get("content") == item:
                new_input.append(item)
            else:
                new_input.append(new_messages[i])
        # Items beyond new_messages length were removed by dedup — skip them

    # Stdout log
    if is_stream:
        print(
            "[ContextPrune] Streaming request — dedup applied to input, streaming response passed through",
            flush=True,
        )
    print(
        f"[ContextPrune/Responses] Q{q} ratio={ratio:.2f} removed={removed}sents saved={saved}tok",
        flush=True,
    )

    # File log
    stat_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "endpoint": "responses",
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
    modified_body["input"] = new_input
    if new_system is not None:
        modified_body["instructions"] = new_system
    elif "instructions" in modified_body and system is None:
        pass  # preserve original (empty/missing) instructions as-is

    # ---- Forward to OpenAI Responses API ---------------------------------- #
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    try:
        if is_stream:
            # Dedup was applied to input; forward deduplicated body as streaming request
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = client.stream(
                    "POST",
                    f"{target_base}/v1/responses",
                    headers=headers,
                    json=modified_body,
                )
                async with upstream as resp:
                    return StreamingResponse(
                        resp.aiter_bytes(),
                        status_code=resp.status_code,
                        headers=dict(resp.headers),
                        media_type=resp.headers.get("content-type", "text/event-stream"),
                    )
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{target_base}/v1/responses",
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
    openai_target: Optional[str] = None,
) -> None:
    """Start the ContextPrune proxy server.

    Args:
        port: Port to listen on. Default 8899.
        threshold: Semantic similarity threshold for deduplication. Default 0.82.
        enable_log: Whether to log stats to ~/.contextprune/stats.jsonl.
        host: Host to bind to. Default 127.0.0.1 (localhost only).
        openai_target: Base URL for OpenAI-compatible target. Default https://api.openai.com.
                       Can also be set via CONTEXTPRUNE_OPENAI_TARGET env var.
    """
    import uvicorn

    global _deduplicator, _enable_logging, _openai_target
    _deduplicator = SemanticDeduplicator(similarity_threshold=threshold)
    _enable_logging = enable_log
    if openai_target:
        _openai_target = openai_target.rstrip("/")

    print(f"[ContextPrune] Proxy starting on http://{host}:{port}", flush=True)
    print(f"[ContextPrune] Threshold={threshold} | Logging={'ON' if enable_log else 'OFF'}", flush=True)
    print(f"[ContextPrune] OpenAI target: {_openai_target}", flush=True)
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
    parser.add_argument(
        "--openai-target",
        default=None,
        help="Base URL for OpenAI-compatible endpoint target (default: https://api.openai.com or CONTEXTPRUNE_OPENAI_TARGET env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    serve(
        port=args.port,
        threshold=args.threshold,
        enable_log=not args.no_log,
        host=args.host,
        openai_target=args.openai_target,
    )
