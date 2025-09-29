#!/usr/bin/env bash
set -Eeuo pipefail

log() { echo -e "[$(date -Iseconds)] $*"; }

# ------------------------ Config via ENV ------------------------
: "${REPO_URL:=https://github.com/daver987/gptoss-120-server.git}"  # REQUIRED: set to your repo
: "${BRANCH:=main}"
: "${REPO_DIR:=/workspace/gptoss-120-server}"
: "${ENGINE:=max}"                                 # max | transformers
: "${USE_CUSTOM_OPS:=0}"                           # default off for BF16
: "${MODEL_ID:=openai/gpt-oss-120b}"               # will be overridden if MOD_MODEL_URL+MODEL_LOCAL_DIR set
: "${MODEL_LOCAL_DIR:=/models/gptoss-20b-bf16}"    # local path for BF16 model
: "${MOD_MODEL_URL:=}"                             # e.g., https://builds.modular.com/models/gpt-oss-20b-BF16/20B
: "${HF_TOKEN:=}"
: "${PORT:=8000}"
: "${MODULAR_CHANNEL:=stable}"                     # stable | nightly
: "${TORCH_CUDA:=cu128}"                           # for ENGINE=transformers only
: "${CUDA_VISIBLE_DEVICES:=0}"

export ENGINE MODEL_ID HF_TOKEN PORT CUDA_VISIBLE_DEVICES USE_CUSTOM_OPS

# ------------------------ OS prerequisites ---------------------
log "Installing prerequisites..."
apt-get update -y
apt-get install -y --no-install-recommends git curl ca-certificates build-essential pkg-config wget
rm -rf /var/lib/apt/lists/*

# ------------------------ Install uv (no conda) ----------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

# ------------------------ Clone or update repo -----------------
mkdir -p "$(dirname "$REPO_DIR")"
if [ -d "$REPO_DIR/.git" ]; then
  log "Updating repo at $REPO_DIR..."
  git -C "$REPO_DIR" fetch --all --tags
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
else
  log "Cloning $REPO_URL to $REPO_DIR..."
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ------------------------ Python venv ---------------------------
uv venv .venv || true
# shellcheck disable=SC1091
source .venv/bin/activate
python -V

# ------------------------ Install Modular MAX + Mojo -----------
if [ "$MODULAR_CHANNEL" = "nightly" ]; then
  log "Installing Modular (nightly) ..."
  uv pip install modular --index-url https://dl.modular.com/public/nightly/python/simple/ --prerelease allow
  log "Installing Mojo (nightly) ..."
  uv pip install "mojo<1.0.0" --index-url https://dl.modular.com/public/nightly/python/simple/ --prerelease allow
else
  log "Installing Modular (stable) ..."
  uv pip install modular --extra-index-url https://modular.gateway.scarf.sh/simple/
  log "Installing Mojo (stable) ..."
  uv pip install mojo --extra-index-url https://modular.gateway.scarf.sh/simple/
fi

max --version || { echo "ERROR: max CLI not found after install."; exit 1; }
mojo --version || { echo "ERROR: mojo CLI not found after install."; exit 1; }

# ------------------------ Python deps --------------------------
if [ "$ENGINE" = "transformers" ]; then
  log "ENGINE=transformers → installing CUDA PyTorch ($TORCH_CUDA) and server deps..."
  uv pip install --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA" torch
  uv pip install -r serve/requirements.txt
else
  log "ENGINE=max → installing server deps (skipping torch)..."
  grep -vi '^torch' serve/requirements.txt > /tmp/reqs-no-torch.txt
  uv pip install -r /tmp/reqs-no-torch.txt
fi

# ------------------------ Optional: download Modular BF16 model ---------------
if [ -n "$MOD_MODEL_URL" ]; then
  log "Downloading Modular BF16 20B build from $MOD_MODEL_URL ..."
  mkdir -p "$MODEL_LOCAL_DIR"
  # Try simple recursive fetch; Modular build servers often expose a directory listing.
  # If it's an archive URL, handle that as well.
  case "$MOD_MODEL_URL" in
    *.tar.gz|*.tgz)
      wget -O /tmp/model.tgz "$MOD_MODEL_URL"
      tar -C "$MODEL_LOCAL_DIR" -xzf /tmp/model.tgz
      ;;
    *.zip)
      wget -O /tmp/model.zip "$MOD_MODEL_URL"
      apt-get update -y && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*
      unzip -o /tmp/model.zip -d "$MODEL_LOCAL_DIR"
      ;;
    *)
      # Directory-style URL; mirror into MODEL_LOCAL_DIR
      wget --recursive --no-parent --no-host-directories --reject "index.html*" \
           --cut-dirs=3 --directory-prefix="$MODEL_LOCAL_DIR" "$MOD_MODEL_URL"/ || true
      ;;
  esac
  export MODEL_ID="$MODEL_LOCAL_DIR"
  log "MODEL_ID set to local directory: $MODEL_ID"
fi

# Set HF cache (optional)
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# ------------------------ Launch server ------------------------
log "Starting server on 0.0.0.0:$PORT (ENGINE=$ENGINE, USE_CUSTOM_OPS=$USE_CUSTOM_OPS, MODEL_ID=$MODEL_ID)"
exec uvicorn serve.responses_server:app --host 0.0.0.0 --port "$PORT"
