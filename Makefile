PYTHON := python -u

export PYTHONPATH := $(PWD)/src:$(PYTHONPATH)

MODEL ?= openai/gpt-oss-20b
DEVICES ?= gpu:0
PORT ?= 8000
RESPONSES_PORT ?= 9000

max-serve:
	@echo "Starting MAX for $(MODEL) on $(DEVICES)"
	# --custom-architectures picks up src/gpt_oss_max/architecture (ARCHITECTURES list)
	# --trust-remote-code is harmless for openai/gpt-oss-* (and often required by HF repos)
	max serve \
	  --model $(MODEL) \
	  --devices $(DEVICES) \
	  --custom-architectures gpt_oss_max.architecture \
	  --trust-remote-code \
	  --port $(PORT)

responses-shim:
	$(PYTHON) server/responses_app.py --upstream http://127.0.0.1:$(PORT) --port $(RESPONSES_PORT)

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +