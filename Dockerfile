# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch wheel (saves ~2.5 GB of CUDA we never use).
RUN pip install --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1+cpu \
    && pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: model warmup ----------
FROM python:3.11-slim AS model-warmup

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers

COPY --from=builder /install /usr/local

# Pre-download embeddings + cross-encoder so first query is fast in prod.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('intfloat/multilingual-e5-base'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')"

# ---------- Stage 3: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers \
    TRANSFORMERS_OFFLINE=0 \
    HOST=0.0.0.0 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=model-warmup /models /models

COPY --chown=app:app . .

RUN chown -R app:app /models

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
