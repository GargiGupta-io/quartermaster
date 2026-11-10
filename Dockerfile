# The service in one container, run as a non-root user. Hosts such as Render set PORT.
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

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "exec uvicorn procurement_agent.api:app --host 0.0.0.0 --port $PORT --no-access-log"]
