#!/usr/bin/env bash
set -euo pipefail

export PORT="${PORT:-8000}"
export MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"
export ENGINE="${ENGINE:-max}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec uvicorn serve.responses_server:app --host 0.0.0.0 --port "$PORT"
