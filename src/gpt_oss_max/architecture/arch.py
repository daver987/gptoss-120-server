from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.architectures.llama3 import weight_adapters
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)
from .model import GptOssModel

gpt_oss_arch = SupportedArchitecture(
    # Name must match the HF config's model class naming convention.
    # GPT‑OSS repos use "gpt_oss" with causal LM; using this canonical name ensures mapping.
    name="GptOssForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,  # non‑MoE kept bf16, MoE weights are U8+E internally
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
    },
    pipeline_model=GptOssModel,
    tokenizer=TextTokenizer,
    rope_type=RopeType.normal,
    weight_adapters={
        # Reuse Llama3 adapters for general tensor plumbing; our MXFP4 module handles MoE internals.
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
)
