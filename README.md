# GPT-OSS 120B Responses Server

This repository implements a Harmony-compliant `/v1/responses` server for GPT-OSS-120B following the build plan in `INSTRUCTIONS.md`. The codebase now supports both MAX (preferred) and Hugging Face Transformers engines and ships custom Mojo MXFP4 kernels ready to be compiled inside the MAX runtime.

Key capabilities:
- **Mojo MXFP4 kernels** for quantize, dequantize, and a fused decode mat-vec (`custom_ops/kernels/mxfp4.mojo`).
- **Engine switcher** (`serve/model_loader.py`) that loads either MAX (`ENGINE=max`) or Transformers (`ENGINE=transformers`) at runtime while sharing the Harmony tokenizer.
- **FastAPI Responses server** (`serve/responses_server.py`) exposing `/v1/responses`, `/healthz`, and `/bench`, preserving tool-calls and structured outputs.
- **Start script & dependencies** (`serve/start.sh`, `serve/requirements.txt`) tuned for MAX NVIDIA containers but runnable in other environments where GPUs are available.
- **Unit tests** (`tests/`) that validate Harmony rendering, messaging, and tool-call parsing via lightweight stubs—no GPU required.

## Repository layout

```
custom_ops/             # Mojo kernels compiled by MAX at runtime
serve/
  model_loader.py       # Engine factory + shared tokenizer
  responses_server.py   # FastAPI app with Responses + bench endpoint
  requirements.txt      # Runtime dependencies (FastAPI, Transformers, torch)
  start.sh              # UVicorn entrypoint (reads ENGINE/MODEL_ID/HF_TOKEN)
Dockerfile              # MAX base image setup (build remotely or on Runpod)
INSTRUCTIONS.md         # Original drop-in guidance
README.md               # This document
AGENTS.md               # Architecture + change log
pixi.toml               # Local dev/test environment definition
tests/                  # Server unit tests using stubbed runtime
```

## Local development (macOS without Docker)

1. Install dependencies into a Python 3.11+ environment (Pixi keeps things reproducible):
   ```bash
   pixi install
   ```
2. Run the unit test suite (uses stub engines/tokenizers, so no GPU is required):
   ```bash
   pixi run test
   ```

>⚠️ The real GPT-OSS-120B weights are far too large for a Mac laptop. Use the tests to validate behaviour locally; perform real inference inside a GPU container (Runpod, etc.).

## Running the server (GPU environment)

Set the relevant environment variables, then launch Uvicorn:

```bash
export MODEL_ID=openai/gpt-oss-120b
export ENGINE=max             # or transformers
# export HF_TOKEN=...         # if weights are gated
uvicorn serve.responses_server:app --host 0.0.0.0 --port 8000
```

- `ENGINE=max` expects the Modular MAX runtime and will load the Mojo kernels automatically.
- `ENGINE=transformers` runs the Hugging Face pipeline (useful if MAX lacks MXFP4 loaders yet).
- `/healthz` reports the active engine; `/bench?n=8&max_new_tokens=64` provides a quick decode TPS probe.

## Preparing for Runpod (build remotely or on the pod)

Although Docker cannot run on this Mac, you can still build the container image remotely or directly on Runpod:

1. **(Remote) Build & push**
   ```bash
   docker build -t <registry>/gptoss-mxfp4:latest .
   docker push <registry>/gptoss-mxfp4:latest
   ```
   *(Run these on a machine with Docker support, e.g., Runpod build pod, remote Linux box, or GitHub Actions.)*

2. **Pod environment variables**
   - `MODEL_ID=openai/gpt-oss-120b`
   - `ENGINE=max` (fall back to `transformers` if MAX cannot load MXFP4 yet)
   - `HF_TOKEN=<your_token>` (if the model is gated)

3. **First boot checks**
   - `curl http://<pod-ip>:8000/healthz`
   - `curl http://<pod-ip>:8000/bench?n=8&max_new_tokens=64`

Continue with the detailed Runpod bring-up instructions from the latest guidance once you are on a GPU pod.

## Next steps

- Implement the tiled MXFP4 qGEMM for prefill (Stage B of the plan) and integrate batching (Stage C).
- Capture `/bench` metrics before and after kernel optimizations to track TPS improvements.
- Follow the Runpod instructions to deploy on an H100 pod once the container image is available.
