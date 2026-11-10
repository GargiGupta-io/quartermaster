# The service as Hugging Face Spaces runs it: one container, port 7860, a non-root user.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app/service

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/home/app/service/.cache/fastembed \
    DATA_DIR=/tmp/procurement-data \
    PATH="/home/app/service/.venv/bin:$PATH"

# Dependencies first, so code changes don't reinstall them.
COPY --chown=app pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app src ./src
COPY --chown=app data ./data
RUN uv sync --frozen --no-dev

# Download the embedding model and build the search index now, not on the first request.
RUN python -c "from procurement_agent.knowledge import KnowledgeBase; print(KnowledgeBase().size, 'passages indexed')"

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
CMD ["uvicorn", "procurement_agent.api:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
