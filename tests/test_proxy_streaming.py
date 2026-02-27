"""Regression tests for proxy streaming.

Before the fix, returning StreamingResponse(resp.aiter_bytes()) from inside
`async with httpx.AsyncClient() ... async with client.stream() as resp` caused
httpx.StreamClosed because the context managers exited (closing the stream)
as soon as the function returned -- before starlette consumed any chunks.

The fix: _make_streaming_response owns the context managers inside the generator
so they stay alive until the last byte is yielded.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_make_streaming_response_consumes_all_chunks():
    """Generator must yield all chunks before closing the stream."""
    from contextprune.proxy import _make_streaming_response

    chunks = [b"data: hello\n\n", b"data: world\n\n", b"data: [DONE]\n\n"]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/event-stream", "content-length": "42"}

    async def fake_aiter_bytes():
        for chunk in chunks:
            yield chunk

    mock_resp.aiter_bytes = fake_aiter_bytes

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_client.aclose = AsyncMock()

    async def run():
        with patch("httpx.AsyncClient", return_value=mock_client):
            streaming_resp = await _make_streaming_response(
                "POST", "http://test/v1/messages", {}, {}
            )
            collected = []
            async for chunk in streaming_resp.body_iterator:
                collected.append(chunk)
        return streaming_resp, collected

    streaming_resp, collected = _run(run())

    assert collected == chunks, "Not all chunks were yielded"
    mock_stream_ctx.__aexit__.assert_called_once()
    mock_client.aclose.assert_called_once()


def test_make_streaming_response_strips_content_length():
    """content-length must not be forwarded -- it causes HPE_UNEXPECTED_CONTENT_LENGTH."""
    from contextprune.proxy import _make_streaming_response

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {
        "content-type": "text/event-stream",
        "content-length": "99",
        "transfer-encoding": "chunked",
        "x-request-id": "abc123",
    }

    async def fake_aiter_bytes():
        yield b"data: ok\n\n"

    mock_resp.aiter_bytes = fake_aiter_bytes

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_client.aclose = AsyncMock()

    async def run():
        with patch("httpx.AsyncClient", return_value=mock_client):
            return await _make_streaming_response(
                "POST", "http://test/v1/messages", {}, {}
            )

    resp = _run(run())

    forwarded = dict(resp.headers)
    assert "content-length" not in forwarded, "content-length must be stripped"
    assert "transfer-encoding" not in forwarded, "transfer-encoding must be stripped"
    assert forwarded.get("x-request-id") == "abc123", "safe headers must be forwarded"


def test_make_streaming_response_preserves_status_code():
    """Upstream status code (e.g. 429, 401) must be forwarded."""
    from contextprune.proxy import _make_streaming_response

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"content-type": "application/json"}

    async def fake_aiter_bytes():
        yield b'{"error":"rate_limit_exceeded"}'

    mock_resp.aiter_bytes = fake_aiter_bytes

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_client.aclose = AsyncMock()

    async def run():
        with patch("httpx.AsyncClient", return_value=mock_client):
            return await _make_streaming_response(
                "POST", "http://test/v1/messages", {}, {}
            )

    resp = _run(run())
    assert resp.status_code == 429
