FROM modular/max-nvidia-full:25.6

WORKDIR /app
COPY custom_ops ./custom_ops
COPY serve ./serve

ENV PYTHONUNBUFFERED=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir -r serve/requirements.txt

ENV MODEL_ID=openai/gpt-oss-120b \
  ENGINE=max \
  HF_HOME=/models/hf_cache \
  CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["cd /app && exec bash serve/start.sh"]
