PYTHON := python -u
export PYTHONPATH := $(PWD)/src:$(PYTHONPATH)

MODEL ?= openai/gpt-oss-20b
DEVICES ?= gpu:0
PORT ?= 8000
RESPONSES_PORT ?= 9000
VENV_DIR ?= .venv
HF_TOKEN ?=

# One-touch: install everything + start both servers
runpod-up:
	@bash scripts/runpod_up.sh

# Explicit start/stop if you prefer separate steps
max-serve:
	@echo "Starting MAX for $(MODEL) on $(DEVICES)"
	max serve \
	  --model $(MODEL) \
	  --devices $(DEVICES) \
	  --custom-architectures gpt_oss_max.architecture \
	  --trust-remote-code \
	  --port $(PORT)

responses-shim:
	$(PYTHON) server/responses_app.py --upstream http://127.0.0.1:$(PORT) --port $(RESPONSES_PORT)

down:
	@bash scripts/stop.sh

logs:
	@echo "--- MAX ---"; tail -n 200 -f logs/max.log

rlogs:
	@echo "--- RESPONSES ---"; tail -n 200 -f logs/responses.log

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .run || true
