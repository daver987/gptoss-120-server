import importlib
import sys
from pathlib import Path

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    module = importlib.import_module("server.responses_app")
    importlib.reload(module)
    return module


@pytest.mark.anyio
async def test_health_endpoint():
    module = _load_module()
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_responses_to_chat_payload_maps_core_fields():
    module = _load_module()
    payload = module.responses_to_chat_payload(
        {
            "model": "openai/gpt-oss-120b",
            "input": [
                {"role": "user", "content": "hello"},
            ],
            "max_output_tokens": 42,
            "response_format": {"type": "json_object"},
            "tools": [{"type": "function", "function": {"name": "noop"}}],
            "temperature": 0.2,
        }
    )
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["messages"][0]["content"] == "hello"
    assert payload["max_tokens"] == 42
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["tools"][0]["function"]["name"] == "noop"
    assert payload["temperature"] == 0.2


def test_chat_to_responses_wraps_choice_payload():
    module = _load_module()
    result = module.chat_to_responses(
        {
            "id": "chatcmpl-123",
            "created": 1700000000,
            "model": "openai/gpt-oss-120b",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Echo reply",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "noop", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
    )
    assert result["object"] == "response"
    assert result["output"][0]["type"] == "message"
    assert result["output"][0]["content"] == "Echo reply"
    assert result["output"][0]["tool_calls"][0]["function"]["name"] == "noop"
    assert result["usage"]["prompt_tokens"] == 12


@pytest.mark.anyio
async def test_responses_endpoint_calls_upstream(monkeypatch):
    module = _load_module()
    captured = {}

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class DummyAsyncClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return DummyResponse(
                {
                    "id": "chatcmpl-123",
                    "created": 1700000000,
                    "model": "openai/gpt-oss-120b",
                    "choices": [
                        {"message": {"role": "assistant", "content": "Echo reply"}}
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 5},
                }
            )

    request = {
        "model": "openai/gpt-oss-120b",
        "input": [{"role": "user", "content": "hello"}],
        "max_output_tokens": 32,
    }
    transport = httpx.ASGITransport(app=module.app)
    original_async_client = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient", DummyAsyncClient)
    async with original_async_client(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post("/v1/responses", json=request)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "openai/gpt-oss-120b"
    assert data["output"][0]["content"] == "Echo reply"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["json"]["messages"][0]["content"] == "hello"
    assert captured["json"]["max_tokens"] == 32
