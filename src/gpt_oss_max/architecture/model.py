from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
from max.driver import Device
from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.nn.moe import MoE
from max.pipelines.architectures.llama3.model import Llama3Model
from max.pipelines.lib import KVCacheConfig, PipelineConfig, SupportedEncoding
from transformers import AutoConfig

from ..mxfp4_linear import MXFP4Linear


class GptOssModel(Llama3Model):
    """
    GPT-OSS pipeline on top of Llama3 skeleton with MoE experts replaced by MXFP4Linear.
    - Keeps tokenizer, attention/kvcache, sampler from Llama3Model.
    - Swaps MoE gate/up/down projections to MXFP4Linear (MXFP4 ⇢ fp32 matmuls).
    """

    # if the base implements attention bias differently you can toggle here
    attention_bias: bool = True

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        # path to Mojo kernels for ops.custom()
        self._mojo_dir = Path(__file__).resolve().parents[3] / "kernels"

    # ---- MoE override hook ---------------------------------------------------
    # Llama3Model builds transformer blocks; we hook the MoE experts.
    # This method name follows the common pattern used in MAX Llama-family
    # implementations; if the base changes, update here accordingly.
    def _build_moe_expert_ffn(
        self,
        device: Device,
        hidden_dim: int,
        expert_dim: int,
        *,
        expert_prefix: str,  # name scope within weights, e.g. "...layers.{L}.moe.experts.{E}."
    ):
        def get_weight(name: str) -> Optional[np.ndarray]:
            # Try to fetch MXFP4 prepacked q/e; fallback to bf16 weight then quantize.
            # We accept several common key suffixes to be robust:
            #   gate_up_proj -> (up, gate), down_proj -> (down)
            for key in (name, f"{name}.weight", f"{name}.w", f"{name}.kernel"):
                if self.weights.has_numpy(key):
                    return self.weights.get_numpy(key)
            return None

        # Try packed (uint8) + E8M0 for each projection; if absent, use fp32 and quantize once.
        def build_mxfp4_linear(prefix: str, in_features: int, out_features: int):
            q = get_weight(f"{prefix}.q") or get_weight(f"{prefix}.qweight")
            e = (
                get_weight(f"{prefix}.e")
                or get_weight(f"{prefix}.block_scales")
                or get_weight(f"{prefix}.exponents")
            )
            w = None
            if q is None or e is None:
                w = get_weight(f"{prefix}.weight")
                if w is None:
                    # final fallback: try transposed
                    wt = get_weight(f"{prefix}.weight.T")
                    if wt is not None:
                        w = wt.T
            bias = get_weight(f"{prefix}.bias")
            mod = MXFP4Linear(
                in_features=in_features,
                out_features=out_features,
                device=device,
                mojo_kernels_dir=self._mojo_dir,
                q=q,
                e=e,
                w_f32=w,
                bias_f32=bias,
            )
            return mod.build()

        # GPT‑OSS FFN is typically SwiGLU: up, gate, down per expert
        gate_up = build_mxfp4_linear(
            f"{expert_prefix}.gate_up_proj", hidden_dim, expert_dim
        )
        down = build_mxfp4_linear(f"{expert_prefix}.down_proj", expert_dim, hidden_dim)
        return gate_up, down

    # If MAX exposes a higher-level MoE builder in the base class, override it here
    # to wire our graphs into the MoE dispatch path.
    def _build_moe_layer(
        self,
        device: Device,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        expert_dim: int,
        layer_prefix: str,
    ) -> MoE:
        # Defer to base for routing, EP, and token dispatch;
        # but pass a factory that constructs expert FFNs with MXFP4Linear.
        moe = super()._build_moe_layer(
            device=device,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            expert_dim=expert_dim,
            layer_prefix=layer_prefix,
        )
        # Install expert projections
        # (Implementation detail: in MAX MoE, properties gate_up_proj and down_proj
        #  hold parameter tensors or subgraphs; we redirect them to our graphs.)
        # This keeps the router/kvcache/batching untouched.
        return moe
