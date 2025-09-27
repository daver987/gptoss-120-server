FROM modular/max-nvidia-full:25.6.0.dev2025090605

WORKDIR /app
COPY custom_ops ./custom_ops
COPY serve ./serve

RUN pip install --no-cache-dir -r serve/requirements.txt

EXPOSE 8000
ENV MODEL_ID="openai/gpt-oss-120b"
CMD ["bash", "serve/start.sh"]
