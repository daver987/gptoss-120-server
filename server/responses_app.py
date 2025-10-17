from __future__ import annotations
import argparse
import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()
UPSTREAM = os.environ.get("MAX_UPSTREAM", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "300"))


def responses_to_chat_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map OpenAI Responses payload to MAX /v1/chat/completions.
    We keep messages intact; Harmony chat template is applied by the model server.
    """
    model = body.get("model")
    # Responses API has "input" that may be messages or content array; accept both.
    msgs = body.get("input") or body.get("messages")
    if isinstance(msgs, dict):
        # single message object
        msgs = [msgs]
    if not isinstance(msgs, list):
        raise ValueError("Expected 'input' or 'messages' as a list of chat messages.")

    out: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
    }
    # map token limits and JSON/structured outputs
    if "max_output_tokens" in body:
        out["max_tokens"] = body["max_output_tokens"]
    if "response_format" in body:
        out["response_format"] = body["response_format"]
    # tools (OpenAI-compatible)
    if "tools" in body:
        out["tools"] = body["tools"]
    if "tool_choice" in body:
        out["tool_choice"] = body["tool_choice"]
    # sampling params (best effort)
    for k_in, k_out in [("temperature", "temperature"), ("top_p", "top_p")]:
        if k_in in body:
            out[k_out] = body[k_in]
    return out


def chat_to_responses(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map MAX /v1/chat/completions result back to a Responses-like envelope.
    """
    # Simply wrap chat.completions payload; tools and choices already encoded.
    return {
        "id": body.get("id"),
        "object": "response",
        "created": body.get("created"),
        "model": body.get("model"),
        "output": [
            {
                "type": "message",
                "role": body["choices"][0]["message"]["role"],
                "content": body["choices"][0]["message"].get("content", ""),
                "tool_calls": body["choices"][0]["message"].get("tool_calls"),
            }
        ],
        "usage": body.get("usage", {}),
    }


@app.post("/v1/responses")
async def responses(request: Request):
    body = await request.json()
    payload = responses_to_chat_payload(body)
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(f"{UPSTREAM}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    return JSONResponse(chat_to_responses(data))


@app.get("/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 9000)))
    args = ap.parse_args()
    os.environ["MAX_UPSTREAM"] = args.upstream
    uvicorn.run("responses_app:app", host="0.0.0.0", port=args.port, reload=False)
