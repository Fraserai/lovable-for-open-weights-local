# Weightroom: one image for both services.
#   web:    the default command (API + MCP server + pages)
#   worker: start command `python -m weightroom.worker`
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
WORKDIR /app/worker

# Dependencies first (cached between deploys). Tinker trains on its own GPUs; torch is only needed here
# for tokenizers and chat formats, so install the much smaller CPU build. The documents group
# (`trainloop prepare`) is parked and left out.
COPY worker/pyproject.toml worker/uv.lock ./
RUN uv sync --frozen --no-dev --no-group documents --no-install-project --no-install-package torch \
 && uv pip install --python .venv/bin/python "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu

COPY worker/ ./
RUN uv sync --frozen --no-dev --no-group documents --no-install-package torch

ENV PATH="/app/worker/.venv/bin:$PATH" PYTHONPATH=/app/worker/src DATA_DIR=/data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn weightroom.api:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
