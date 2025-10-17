#!/usr/bin/env bash
# One-touch bootstrap for Runpod (or any GPU VM).
# - Creates a venv with uv
# - Installs Modular MAX & Mojo toolchain
# - Installs this package (OpenAI Harmony + shim deps)
# - Starts MAX OpenAI-compatible server (pulls GPT-OSS automatically)
# - Starts /v1/responses shim
set -euo pipefail

### ---------- Configuration (override via env) ----------
MODEL="${MODEL:-openai/gpt-oss-20b}"   # or openai/gpt-oss-120b if you have 80GB VRAM
DEVICES="${DEVICES:-gpu:0}"            # e.g., gpu:0 or gpu:all
PORT="${PORT:-8000}"                   # MAX server
RESPONSES_PORT="${RESPONSES_PORT:-9000}" # Harmony Responses shim
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"
HF_TOKEN="${HF_TOKEN:-}"               # optional; GPT-OSS is public, but token helps rate limiting
EXTRA_PIP_ARGS="${EXTRA_PIP_ARGS:-}"   # for proxies, etc.

# Speed up HF downloads if hf_transfer present
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

### ---------- Helpers ----------
log() { echo -e "\033[1;32m[runpod]\033[0m $*"; }
err() { echo -e "\033[1;31m[runpod]\033[0m $*" >&2; }

### ---------- Sanity ----------
if ! command -v curl >/dev/null 2>&1; then
  err "curl not found. apt-get update && apt-get install -y curl"
  exit 1
fi

# Basic GPU presence check (NVIDIA / AMD)
if command -v nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
elif command -v rocminfo >/dev/null 2>&1; then
  log "AMD GPU detected"
else
  err "No GPU tools found (nvidia-smi/rocminfo). Ensure your Runpod image exposes the GPU."
fi

### ---------- uv & virtualenv ----------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv (fast Python installer)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

log "Creating venv: ${VENV_DIR}"
uv venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

### ---------- Install Modular MAX + Mojo ----------
# Prefer stable channel; fallback to nightly if needed
if ! ${PYTHON} - <<'PY' >/dev/null 2>&1
import importlib; importlib.import_module("max")
PY
then
  log "Installing Modular (MAX + Mojo) [stable]"
  uv pip install modular --extra-index-url https://modular.gateway.scarf.sh/simple/ ${EXTRA_PIP_ARGS} || {
    log "Stable failed; installing Modular (nightly)"
    uv pip install modular --index-url https://dl.modular.com/public/nightly/python/simple/ --prerelease allow ${EXTRA_PIP_ARGS}
  }
else
  log "Modular already present"
fi

# Show max version
log "MAX CLI:"
max --version || true

### ---------- Optional: Hugging Face auth / accelerated downloads ----------
uv pip install "huggingface_hub[hf_transfer]" hf-transfer ${EXTRA_PIP_ARGS}
if [[ -n "${HF_TOKEN}" ]]; then
  log "Logging in to Hugging Face with provided token"
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential --stderr
fi

### ---------- Project install ----------
log "Installing project (shim + harmony)"
uv pip install -e . openai-harmony fastapi uvicorn httpx numpy ${EXTRA_PIP_ARGS}

mkdir -p logs .run

### ---------- Start MAX server ----------
log "Starting MAX server on :${PORT} (model=${MODEL}, devices=${DEVICES})"
# --custom-architectures ensures our MXFP4 architecture is loaded
# --trust-remote-code allows HF configs with custom fields
nohup max serve \
  --model "${MODEL}" \
  --devices "${DEVICES}" \
  --custom-architectures gpt_oss_max.architecture \
  --trust-remote-code \
  --port "${PORT}" \
  > logs/max.log 2>&1 &

echo $! > .run/max.pid

# Wait for /health
log "Waiting for MAX to come up..."
log "MAX is now downloading weights and compiling graphs; this step can take a few minutes."
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    log "MAX is healthy on :${PORT}"
    break
  fi
  if (( i % 15 == 0 )); then
    last_log="$(tail -n 1 logs/max.log 2>/dev/null || echo 'no log yet')"
    log "MAX still starting (attempt ${i}/240). Latest max.log: ${last_log}"
  fi
  sleep 2
  if ! kill -0 "$(cat .run/max.pid 2>/dev/null || echo 0)" 2>/dev/null; then
    err "MAX process exited unexpectedly. Check logs/max.log"
    exit 1
  fi
done
if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
  err "MAX never became healthy; exiting."
  exit 1
fi

### ---------- Start Responses shim ----------
log "Starting /v1/responses shim on :${RESPONSES_PORT}"
nohup "${PYTHON}" server/responses_app.py \
  --upstream "http://127.0.0.1:${PORT}" \
  --port "${RESPONSES_PORT}" \
  > logs/responses.log 2>&1 &

echo $! > .run/responses.pid

log "Waiting for Responses shim health on :${RESPONSES_PORT}"
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${RESPONSES_PORT}/health" >/dev/null; then
    log "Responses shim is healthy on :${RESPONSES_PORT}"
    break
  fi
  if (( i % 10 == 0 )); then
    last_log="$(tail -n 1 logs/responses.log 2>/dev/null || echo 'no log yet')"
    log "Responses shim still starting (attempt ${i}/120). Latest responses.log: ${last_log}"
  fi
  sleep 1
  if ! kill -0 "$(cat .run/responses.pid 2>/dev/null || echo 0)" 2>/dev/null; then
    err "Responses shim exited unexpectedly. Check logs/responses.log"
    exit 1
  fi
done
if ! curl -fsS "http://127.0.0.1:${RESPONSES_PORT}/health" >/dev/null; then
  err "Responses shim never became healthy; exiting."
  exit 1
fi

log "-------------------------------------------------------------------"
log " Ready!"
log "  MAX (OpenAI-compatible):  http://127.0.0.1:${PORT}/v1"
log "  Responses API (Harmony):   http://127.0.0.1:${RESPONSES_PORT}/v1/responses"
log "  Logs: tail -f logs/max.log | logs/responses.log"
log "-------------------------------------------------------------------"
