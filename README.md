# SIRCA-RAG

**Corrective RAG Híbrido para Generación Fundamentada sobre Plantas Medicinales Peruanas**

> Sistema de soporte para el paper enviado a **SimBig 2025 / WAIMLAp 2026 / SIPAIM**.
> Repositorio oficial de implementación, evaluación y deployment.

## What is this?

A Corrective RAG (CRAG) system that answers questions about Peruvian medicinal plants using scientific literature. It retrieves evidence from a curated corpus, evaluates relevance, and generates grounded answers with citations — in Spanish or English.

**Key contributions:**
1. A hybrid retrieval pipeline (dense + sparse + cross-encoder reranking) with corrective evaluation that improves answer faithfulness over vanilla RAG in the ethnobotanical domain.
2. A reproducible **honest Recall** methodology using pool-based ground truth (avoids the inflated `Recall=1.0` artifact seen in self-judging RAG evaluations).
3. An embedding fine-tuning experiment that lifts **Top-5 accuracy on vernacular Quechua / market / regional plant names from 0.25 to 0.80**, demonstrating linguistic resilience to native names absent in the base corpus.

## Architecture

```
Query -> Classify -> Hybrid Retrieve -> CRAG Evaluate -> Generate -> Answer + Citations
```

| Stage | Component | Details |
|-------|-----------|---------|
| **Classify** | Rule-based classifier | factual (α=0.3) / exploratory (α=0.7) / comparative (α=0.5) |
| **Retrieve** | Dense (e5-multilingual 768d) + BM25 sparse | RRF fusion, top-30 candidates, reranked to top-10 with `ms-marco-MiniLM-L-12-v2` |
| **Evaluate** | CRAG threshold | accept (>=0.35), refine (0.15-0.35), web_search (<0.15) |
| **Generate** | Chain-of-Thought grounding | Extract → Verify → Compose. DeepSeek V4 Flash, Ollama (Qwen3.5), or template backend |

Tuning (**Plan A**) applied: `RETRIEVAL_TOP_K=30`, `RERANK_TOP_K=10`, per-category α.

## Evaluation Results — Plan A (DeepSeek backend)

Honest Recall via pool-based ground truth (POOL_K=30):

| Metric | Score |
|---|---|
| Recall (honest) | **0.568** |
| MRR | **0.873** |
| NDCG@10 | **0.845** |
| Faithfulness | **0.611** |
| BERTScore F1 (roberta-large) | 0.901 |
| Semantic Similarity | 0.991 |

### Vernacular-name resistance (fine-tuning side experiment)

20-query benchmark using **only** common, market, Quechua, Aymara and regional names:

| Model | Mean target similarity | Top-1 | Top-5 |
|---|---|---|---|
| `e5-base` (baseline) | 0.20 | 0.25 | 0.50 |
| `e5-base + FT vernacular` | **0.40** | **0.80** | **0.88** |

Trained with `MultipleNegativesRankingLoss` over 573 contrastive triplets including Quechua aliases (`muña muña`, `huamanpinta`, `wamanripa`, `yawar wayqu`, `ayuk willku`, …). See `finetuning/` for the reproducible pipeline.

## Data Corpus

- **100 medicinal plant species** from southern Peru (Arequipa, Cusco, Puno, Moquegua, Tacna)
- **16,486 unique articles** from 4 literature APIs: PubMed, Europe PMC, Semantic Scholar, CrossRef
- **Structured data** from GBIF (occurrences), PeruNPDB (phytochemicals), WFO (taxonomy), COCONUT (natural products)
- **6,296 vectorized chunks** (balanced 100/species, 91 species with literature)
- **Embedding**: `intfloat/multilingual-e5-base` (768 dimensions)
- **Chunking**: 512 tokens, 64 overlap

## Quick Start (development)

```bash
# Install
pip install -r requirements.txt

# Check corpus status
python pipeline.py status

# Re-run vectorization (if needed)
python pipeline.py vectorize

# Start web service
python pipeline.py serve
# Open http://localhost:8000

# Run evaluation
python run_evaluation.py --backend deepseek

# Run ablation study
python run_evaluation.py --ablation --backend deepseek
```

## Production Deployment (Docker)

Multi-stage Dockerfile that:
- Installs `torch` from the CPU index (saves ~2.5 GB of unused CUDA).
- Pre-downloads `multilingual-e5-base` and the cross-encoder at build time, so the **first query is fast**.
- Runs as non-root user `app`.
- Exposes port `8000` with a `curl`-based healthcheck on `/api/health`.

```bash
# Configure
cp env.example .env
# Edit .env -> set DEEPSEEK_API_KEY

# Build + run
docker compose build
docker compose up -d

# Verify
curl http://localhost:8000/api/health
docker compose logs -f sirca-rag
```

`docker-compose.yml` ships with resource limits (4 GB RAM / 2 CPU), log rotation (10 MB × 3), and `restart: unless-stopped`. Ready for Dokploy / Render / any Compose-compatible host.

## Environment Variables

```bash
# Required for LLM generation (DeepSeek backend)
DEEPSEEK_API_KEY=your-key-here

# Optional
OLLAMA_BASE_URL=http://localhost:11434   # For local Ollama backend
HOST=0.0.0.0                             # Web server host
PORT=8000                                # Web server port
HF_HOME=/models                          # HuggingFace model cache (inside Docker)
```

## Pipeline Commands

```bash
python pipeline.py acquire    # Fetch from all 7 sources (resilient, per-species)
python pipeline.py chunk      # Deduplicate + chunk raw articles
python pipeline.py vectorize  # Encode chunks -> FAISS + BM25 indexes
python pipeline.py serve      # Start FastAPI web service
python pipeline.py all        # acquire + chunk + vectorize
python pipeline.py status     # Show corpus and vectorstore stats
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive dark-themed frontend |
| `GET` | `/api/health` | System status, backends, vectorstore size |
| `POST` | `/api/query` | Full pipeline: `{query, backend, language}` -> answer + citations + trace |
| `GET` | `/api/species` | 100 species catalog with department filter |
| `GET` | `/api/species/{name}` | Species detail (common names, departments, uses) |

## Project Structure

```
sirca_rag/
  pipeline.py                # Single entry point for all operations
  run_evaluation.py          # Benchmark runner
  run_validate_all.py        # Environment health check

  agent/                     # CRAG agent pipeline
    graph.py                 # SIRCAAgent orchestrator
    crag_evaluator.py        # Accept/refine/web_search decision
    query_classifier.py      # Factual/exploratory/comparative

  config/settings.py         # Species catalog, search queries, all config
  evaluation/                # BERTScore, faithfulness, retrieval metrics, ablation
  generation/                # Grounded generator (DeepSeek/Ollama/template)
  ingestion/                 # Data source clients (PubMed, EPMC, S2, CrossRef, GBIF, PeruNPDB, WFO, COCONUT)
  retrieval/                 # Hybrid retriever (FAISS + BM25 + cross-encoder)
  scraping/                  # CRAG web search fallback
  web/                       # FastAPI service + dark-themed frontend
  finetuning/                # Vernacular-resistance experiment (triplet contrastive learning)

  Dockerfile                 # Multi-stage production image
  docker-compose.yml         # Compose deployment with healthcheck + limits
  .dockerignore              # Excludes paper, evaluations, FT artifacts

  data/                      # gitignored — regenerable via pipeline.py
    raw/                     # Source-specific JSON files
    processed/               # chunks_expanded.json
    vectorstore/             # index.faiss + metadata.json + bm25_index.pkl
```

## Backends

| Backend | Type | Speed | Quality | Requires |
|---------|------|-------|---------|----------|
| DeepSeek V4 Flash | API | ~30-150s | Best | `DEEPSEEK_API_KEY` |
| Ollama (Qwen3.5) | Local | ~10-30s | Good | Ollama running |
| Template | Passthrough | < 1ms | Test only | Nothing |

## Ablation Study (6 configurations)

| Config | Description |
|--------|-------------|
| `full` | Complete pipeline (baseline) |
| `dense_only` | FAISS dense retrieval only (alpha=1.0) |
| `sparse_only` | BM25 sparse retrieval only (alpha=0.0) |
| `no_reranker` | Hybrid without cross-encoder reranking |
| `no_crag` | Skip CRAG evaluation, always accept |
| `no_classifier` | No query classification, default alpha |

```bash
python run_evaluation.py --ablation --backend deepseek
python run_evaluation.py --ablation --configs full dense_only sparse_only
```

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Chain-of-Thought grounding | 3-step protocol (Extract, Verify, Compose) prevents hallucination |
| Hybrid retrieval (dense + sparse) | Dense catches semantics, sparse catches exact terms (species names, compounds) |
| Cross-encoder reranking | Deep query-document interaction improves precision over bi-encoder alone |
| CRAG evaluation | Threshold-based decision avoids generating from low-quality context |
| Honest Recall via pool-based GT | Avoids inflated `Recall=1.0` from self-judging |
| Balanced vectorization (100/species) | Prevents popular species from dominating retrieval results |
| `multilingual-e5-base` | Bilingual EN/ES support, practical CPU encoding speed |
| Per-species resilient acquisition | One API failure doesn't kill the entire batch |
| Vernacular FT (side experiment) | Demonstrates a low-cost path to Quechua/regional name resistance without re-indexing the corpus |

---

*Built for SimBig 2025 / WAIMLAp 2026 / SIPAIM.*
