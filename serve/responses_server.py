"""FastAPI server exposing a /v1/responses endpoint backed by MAX LLM.generate."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model_loader import ENGINE as ENGINE_NAME
from .model_loader import MODEL_ID, build_engine_and_tokenizer


# -------------------------- Pydantic models ---------------------------------
class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: Dict[str, Any]


class MessageContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]


class ResponsesRequest(BaseModel):
    model: Optional[str] = None
    input: List[Message]
    tools: Optional[List[ToolSpec]] = None
    response_format: Optional[ResponseFormatJSONSchema] = None
    max_output_tokens: Optional[int] = 1024


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, Any]


class ResponsesOutput(BaseModel):
    type: str
    content: Optional[List[MessageContent]] = None
    tool_calls: Optional[List[ToolCall]] = None


class ResponsesResponse(BaseModel):
    id: str
    model: str
    output: List[ResponsesOutput]
    status: str = "completed"
    usage: Optional[Dict[str, int]] = None


# -------------------------- Runtime helpers ---------------------------------
@lru_cache(maxsize=1)
def _get_runtime() -> Tuple[Any, Any]:
    return build_engine_and_tokenizer()


def _normalize_contents(msg: Message) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(part.text for part in msg.content)


def _render_harmony(
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None,
    response_format: ResponseFormatJSONSchema | None,
) -> str:
    _, tokenizer = _get_runtime()

    rendered_msgs: List[Dict[str, Any]] = []
    if tools:
        tool_desc = {
            "type": "tools",
            "spec": [t.function.model_dump() for t in tools],
        }
        rendered_msgs.append({
            "role": "developer",
            "content": json.dumps(tool_desc, ensure_ascii=False),
        })
    if response_format:
        fmt_desc = {"type": "response_format", **response_format.model_dump()}
        rendered_msgs.append({
            "role": "developer",
            "content": json.dumps(fmt_desc, ensure_ascii=False),
        })

    rendered_msgs.extend(
        {"role": m.role, "content": _normalize_contents(m)} for m in messages
    )

    prompt_text = tokenizer.apply_chat_template(
        rendered_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text


TOOL_CALL_RE = re.compile(
    r"<\|start\|>assistant<\|channel\|>commentary\s+to=functions\.([a-zA-Z0-9_]+).*?<\|message\|>(\{.*?\})",
    re.DOTALL,
)
FINAL_MESSAGE_RE = re.compile(
    r"<\|start\|>assistant(?:<\|channel\|>[a-zA-Z0-9_]+)?<\|message\|>(.*)",
    re.DOTALL,
)


def _parse_output(text: str) -> ResponsesOutput:
    match = TOOL_CALL_RE.search(text)
    if match:
        fn_name = match.group(1)
        raw_payload = match.group(2)
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            payload = {"$raw": raw_payload}
        return ResponsesOutput(
            type="tool_call",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function={"name": fn_name, "arguments": payload},
                )
            ],
        )

    final_match = FINAL_MESSAGE_RE.search(text)
    if final_match:
        cleaned = final_match.group(1).strip()
    else:
        cleaned = re.sub(r"<\|.*?\|>", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = text.strip()
    return ResponsesOutput(
        type="message",
        content=[MessageContent(text=cleaned)],
    )


app = FastAPI()


@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"ok": True, "engine": ENGINE_NAME}


@app.post("/v1/responses", response_model=ResponsesResponse)
def responses(req: ResponsesRequest) -> ResponsesResponse:
    if not req.input:
        raise HTTPException(status_code=400, detail="input must contain at least one message")

    engine, tokenizer = _get_runtime()
    prompt = _render_harmony(req.input, req.tools or [], req.response_format)
    max_new = int(req.max_output_tokens or 1024)

    t0 = time.time()
    try:
        completions = engine.generate([prompt], max_new_tokens=max_new)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    dt = time.time() - t0
    if not completions:
        raise HTTPException(status_code=500, detail="LLM returned no outputs")

    out_text = completions[0]

    tokens_in = len(tokenizer.encode(prompt))
    tokens_out = len(tokenizer.encode(out_text))
    tps = int(tokens_out / dt) if dt > 0 else 0

    output = _parse_output(out_text)
    response_id = f"resp_{uuid.uuid4().hex}"

    return ResponsesResponse(
        id=response_id,
        model=req.model or MODEL_ID,
        output=[output],
        usage={
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "tps": tps,
        },
    )


@app.get("/bench")
def bench(n: int = 4, max_new_tokens: int = 64) -> Dict[str, Any]:
    engine, tokenizer = _get_runtime()
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Summarize MXFP4 in 3 concise bullets.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    t0 = time.time()
    outputs = engine.generate([prompt] * max(1, n), max_new_tokens=max_new_tokens)
    dt = max(1e-6, time.time() - t0)
    total_tokens = sum(len(tokenizer.encode(o)) for o in outputs)
    return {
        "requests": max(1, n),
        "max_new_tokens": max_new_tokens,
        "total_output_tokens": total_tokens,
        "wall_seconds": round(dt, 3),
        "aggregate_tps": round(total_tokens / dt, 1) if dt > 0 else 0.0,
    }


__all__ = ["app", "responses", "health"]
