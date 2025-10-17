# max-gpt-oss-mxfp4

Run **OpenAI gpt-oss** on **Modular MAX** with **custom MXFP4 kernels** (Mojo) for MoE FFN linears, and serve via an **OpenAI Responses API** shim that preserves Harmony.

### Prereqs

- Linux or macOS with a recent NVIDIA/AMD GPU and drivers.
- Python 3.10+ (recommend `uv` or `pixi`)
- Modular MAX + Mojo installed (see docs).
  - `uv pip install modular --extra-index-url https://modular.gateway.scarf.sh/simple/`
  - Or use conda/pixi as in docs.
- Hugging Face CLI (`pip install huggingface_hub`) **optional** – the `max` CLI will pull weights too.

> References: MAX OpenAI server and custom architectures/ops.  
> GPT‑OSS + Harmony format.

### Quickstart (one terminal runs MAX, one runs the Responses shim)

```bash
# 0) clone
git clone https://github.com/you/max-gpt-oss-mxfp4.git
cd max-gpt-oss-mxfp4

# 1) python deps (fastapi shim, harmony helpers)
uv venv && source .venv/bin/activate
uv pip install -e .

# 2) run MAX server with custom architecture (loads Mojo kernels at graph build)
#    Choose one model: openai/gpt-oss-20b (fits ~16GB) or /120b (fits 80GB class)
make max-serve MODEL=openai/gpt-oss-20b DEVICES=gpu:0

# 3) in a new terminal, run the /v1/responses shim that maps to MAX’s /v1/chat/completions
make responses-shim
```

## Test

### Responses API-style request with Harmony role/messages; tools are passed through.

In another terminal, run:

```bash
curl -s http://localhost:9000/v1/responses \
  -H 'content-type: application/json' \
  -d '{
        "model": "openai/gpt-oss-20b",
        "input": [
          {"role":"system","content":"You are helpful."},
          {"role":"user","content":"Explain MXFP4 in 2 bullets."}
        ],
        "max_output_tokens": 128
      }' | jq
```

### Notes

Harmony: gpt-oss expects Harmony formatting; MAX will apply the model’s HF chat_template.jinja automatically, and the shim keeps Responses semantics while forwarding messages intact.

Performance: MoE FFN linears use the fused MXFP4 dequantized matvec/matmul in Mojo; attention and non‑MoE bits use MAX’s highly optimized kernels.

CPU: The kernels have CPU fallbacks but this repo targets GPU.

### Make targets

make max-serve MODEL=openai/gpt-oss-20b DEVICES=gpu:0 # start MAX OpenAI server with custom arch
make responses-shim # start /v1/responses adapter on :9000
make clean # remove py cache
