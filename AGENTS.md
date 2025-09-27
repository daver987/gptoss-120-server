# Agent Handoff Summary

## Current State
- **Custom Mojo kernels** (`custom_ops/kernels/mxfp4.mojo`) implement MXFP4 quantize/dequantize plus a fused GPU decode mat-vec. Helpers are rewritten to avoid Mojo import gaps and expose GPU execution parameters.
- **Engine abstraction** (`serve/model_loader.py`) selects between Modular MAX (`ENGINE=max`) and Hugging Face Transformers (`ENGINE=transformers`). The shared tokenizer is loaded lazily to render Harmony prompts via the model’s `chat_template`.
- **FastAPI service** (`serve/responses_server.py`) now:
  - Renders Harmony with developer messages for tools/structured formats.
  - Provides `/v1/responses`, `/healthz`, and `/bench` endpoints.
  - Parses tool calls vs. final assistant messages while estimating decode TPS.
- **Runtime scripts** (`serve/start.sh`, `serve/requirements.txt`) expose the engine switch at launch and include `torch` so the Transformers fallback works inside the MAX container.
- **Tests** (`tests/test_responses_server.py`) stub the engine/tokenizer to keep local CI lightweight; they cover health, plain responses, and tool-call parsing.
- **Docs** (`README.md`) describe local dev without Docker, engine selection, and the steps required before Runpod deployment.

## Architecture Overview
1. **Client request** (Harmony messages, optional tools/response_format) hits `/v1/responses`.
2. **Harmony rendering** uses the shared tokenizer from `serve/model_loader.py` to apply the HF chat template, ensuring GPT-OSS stays aligned with Harmony expectations.
3. **Engine execution**
   - `ENGINE=max`: MAX loads the custom Mojo kernels via `KernelLibrary` and executes `LLM.generate` offline.
   - `ENGINE=transformers`: Hugging Face pipeline runs decode; intended as a fallback when MAX cannot use MXFP4 yet.
4. **Output parsing** distinguishes tool calls from final assistant text and returns Responses-shaped JSON alongside token usage/TPS telemetry.
5. **Bench endpoint** synthesizes multiple prompts to produce aggregate decode TPS, helping guide kernel and batching optimizations.

## Validation
- `pixi run test` — unit tests using stubbed runtime (no GPU) pass.
- Import-time side effects (heavy tokenizer/engine loads) are avoided until `_get_runtime()` is first invoked, enabling tests to monkeypatch easily.

## Outstanding / Next Steps (before Runpod deployment)
1. Build the container image on a Docker-capable host (or directly on Runpod) using the provided `Dockerfile`.
2. Launch an H100-80GB pod with the exported image, set `MODEL_ID`, `ENGINE`, and (if needed) `HF_TOKEN`, then run smoke tests (`/healthz`, `/bench`, `/v1/responses`).
3. Proceed with performance roadmap:
   - Optimize the Mojo mat-vec kernel (shared-memory reduction, vectorized nibble decode).
   - Implement tiled MXFP4 qGEMM for prefill.
   - Add request batching/streaming as needed for throughput.

## Useful Pointers
- Engine selection/environment variables: `MODEL_ID`, `ENGINE`, `HF_TOKEN`, `PORT` in `serve/start.sh`.
- Hooks for additional telemetry: extend `/bench` or augment `usage` in `responses`.
- Harmonized developer messages (`tools`, `response_format`) live in `render_harmony`—extend here for additional system prompts if future instructions require.
