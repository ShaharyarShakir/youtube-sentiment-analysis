FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PYTHON=/usr/local/bin/python3

RUN uv sync --frozen --no-cache --no-install-project

COPY fastapi-app/ ./fastapi-app/
COPY tfidf_vectorizer.pkl ./
COPY lgbm_model.pkl ./

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "fastapi-app.main:app", "--host", "0.0.0.0", "--port", "8000"]