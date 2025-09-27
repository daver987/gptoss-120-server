"""Engine builder for GPT-OSS 120B with optional MAX or Transformers backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

from transformers import AutoTokenizer

ENGINE = os.getenv("ENGINE", "max").lower()
MODEL_ID = os.getenv("MODEL_ID", "openai/gpt-oss-120b")
HF_TOKEN = os.getenv("HF_TOKEN")

_CUSTOM_OPS_DIR = Path(__file__).resolve().parents[1] / "custom_ops"
_TOKENIZER: AutoTokenizer | None = None


def _load_tokenizer() -> AutoTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        tok_kwargs: dict[str, Any] = {"use_fast": True}
        if HF_TOKEN:
            tok_kwargs["token"] = HF_TOKEN
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, **tok_kwargs)
    return _TOKENIZER


def _build_max_engine() -> Any:
    from max.engine import InferenceSession, devices
    from max.graph import KernelLibrary
    from max.entrypoints.llm import LLM
    from max.pipelines import PipelineConfig

    klib = KernelLibrary.load_paths(context=None, custom_extensions=[_CUSTOM_OPS_DIR])
    session = InferenceSession(devices=[devices.CUDA(0)], custom_extensions=klib)

    pconf = PipelineConfig(model_path=MODEL_ID, dtype="auto")
    llm = LLM(pconf, session=session)

    class MaxEngine:
        def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
            return llm.generate(prompts, max_new_tokens=max_new_tokens)

    return MaxEngine()


def _build_transformers_engine() -> Any:
    import torch
    from transformers import pipeline

    pipe = pipeline(
        task="text-generation",
        model=MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    class TFEngine:
        def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
            outputs: list[str] = []
            for prompt in prompts:
                result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
                if isinstance(result, list) and result:
                    if isinstance(result[0], dict) and "generated_text" in result[0]:
                        outputs.append(result[0]["generated_text"])
                    else:
                        outputs.append(str(result[0]))
                elif isinstance(result, dict) and "generated_text" in result:
                    outputs.append(result["generated_text"])
                else:
                    outputs.append(str(result))
            return outputs

    return TFEngine()


def build_engine_and_tokenizer() -> Tuple[Any, AutoTokenizer]:
    if ENGINE == "transformers":
        engine = _build_transformers_engine()
    elif ENGINE == "max":
        engine = _build_max_engine()
    else:
        raise ValueError(f"Unsupported ENGINE '{ENGINE}'. Use 'max' or 'transformers'.")

    tokenizer = _load_tokenizer()
    return engine, tokenizer


__all__ = ["build_engine_and_tokenizer", "ENGINE", "MODEL_ID", "HF_TOKEN"]
