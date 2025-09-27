import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\nASSISTANT:"
        return rendered

    def encode(self, text):
        return list(range(len(text) // 5 + 1))


class EchoEngine:
    def generate(self, prompts, max_new_tokens=1024):
        return [
            "<|start|>assistant<|channel|>final<|message|>Echo: " + (prompts[0] if prompts else "")
        ]


def _load_module(monkeypatch):
    module = importlib.import_module("serve.responses_server")
    importlib.reload(module)
    monkeypatch.setattr(module, "_get_runtime", lambda: (EchoEngine(), DummyTokenizer()))
    return module


def test_health_endpoint(monkeypatch):
    module = _load_module(monkeypatch)
    assert module.health() == {"ok": True, "engine": "max"}


def test_basic_response_echo(monkeypatch):
    module = _load_module(monkeypatch)
    req = module.ResponsesRequest(
        model="openai/gpt-oss-120b",
        input=[module.Message(role="user", content="hello")],
    )
    resp = module.responses(req)
    assert resp.model == "openai/gpt-oss-120b"
    assert resp.output[0].type == "message"
    content = resp.output[0].content[0].text
    assert content.startswith("Echo:")


def test_tool_call_parsing(monkeypatch):
    module = _load_module(monkeypatch)

    class DummyLLM:
        def generate(self, prompts, max_new_tokens=1024):
            return [
                "<|start|>assistant<|channel|>commentary to=functions.test_tool "
                "<|message|>{\"value\": 1}"
            ]

    monkeypatch.setattr(module, "_get_runtime", lambda: (DummyLLM(), DummyTokenizer()))

    req = module.ResponsesRequest(
        model="openai/gpt-oss-120b",
        input=[module.Message(role="user", content="use the tool")],
        tools=[
            module.ToolSpec(
                function=module.ToolFunction(
                    name="test_tool",
                    description="demo",
                    parameters={"type": "object"},
                )
            )
        ],
    )
    resp = module.responses(req)
    output = resp.output[0]
    assert output.type == "tool_call"
    call = output.tool_calls[0]
    assert call.function["name"] == "test_tool"
    assert call.function["arguments"]["value"] == 1
