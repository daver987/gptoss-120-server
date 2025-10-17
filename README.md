# max-gpt-oss-mxfp4

Run OpenAI gpt-oss on Modular MAX with custom MXFP4 kernels (Mojo) for MoE FFN linears, and serve via an OpenAI Responses API shim that preserves Harmony formatting.

## Prereqs

- Linux or macOS with a recent NVIDIA/AMD GPU and drivers
- Python 3.11+ (recommend uv or pixi)
- Modular MAX + Mojo installed (see docs)
  - uv pip install modular --extra-index-url https://modular.gateway.scarf.sh/simple/
- Hugging Face CLI (optional; max can also pull weights)
  - pip install "huggingface_hub[hf_transfer]" hf-transfer

References: MAX OpenAI server and custom architectures/ops. GPT‑OSS + Harmony format.

## Quickstart (one terminal runs MAX, one runs the Responses shim)

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

## Run on Runpod in one command

```bash
git clone https://github.com/you/max-gpt-oss-mxfp4.git
cd max-gpt-oss-mxfp4

# Optional: choose model / devices
export MODEL=openai/gpt-oss-20b           # or openai/gpt-oss-120b if you have 80GB
export DEVICES=gpu:0                      # or gpu:all

# Optional: HF token to avoid rate limits
# export HF_TOKEN=hf_xxx

./scripts/runpod_up.sh
# or: make runpod-up
```

## Endpoints

- MAX (OpenAI‑compatible): http://<pod-ip>:8000/v1
  - Supports /v1/chat/completions, /v1/completions, /v1/embeddings
- Harmony Responses shim: http://<pod-ip>:9000/v1/responses
  - Preserves Harmony messages; passes tools/JSON through to MAX

## Why this works

- MAX’s server is OpenAI‑compatible by design.
- The graph loads custom Mojo ops via our custom architecture, so the MXFP4 fused dequant matvec/matmul runs inside MAX for MoE FFN linears.
- GPT‑OSS requires Harmony formatting; the shim preserves the Responses‑API contract while MAX applies the model’s chat template.

## Troubleshooting

- Model download slow?
  - We enable hf_transfer by default for faster pulls; ensure HF_HUB_ENABLE_HF_TRANSFER=1 is set.
  - You can also set HF_TOKEN to reduce rate limiting.
- Different GPU setup?
  - Set DEVICES=gpu:all to use all visible GPUs (or stick to gpu:0).
- Ports:
  - Change PORT/RESPONSES_PORT env vars if needed before starting.
- Stop everything:
  - ./scripts/stop.sh or make down
- Logs:
  - tail -f logs/max.log and logs/responses.log
- Health checks:
  - curl -s http://localhost:8000/health
  - curl -s http://localhost:9000/health

## Minimal curl test

```bash
curl -s http://localhost:9000/v1/responses \
  -H 'content-type: application/json' \
  -d '{
        "model": "openai/gpt-oss-20b",
        "input": [
          {"role":"system","content":"You are helpful."},
          {"role":"user","content":"In two bullets, what is MXFP4?"}
        ],
        "max_output_tokens": 64
      }' | jq
```

## Make targets

- make max-serve MODEL=openai/gpt-oss-20b DEVICES=gpu:0  # start MAX OpenAI server with custom arch
- make responses-shim                                    # start /v1/responses adapter on :9000
- make runpod-up                                         # one-touch: install + start both servers
- make down                                              # stop both
- make clean                                             # remove py cache