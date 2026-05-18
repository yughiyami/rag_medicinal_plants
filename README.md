# SIRCA-RAG

**Semi-Autonomous RAG for Peruvian Medicinal Plant Knowledge Integration**

SimBig / WAIMLAp 2026

## Architecture

```
Query → Classify → Retrieve → CRAG Evaluate → Generate
          │        (BGE-M3 +     │              (CoT
          │         BM25 +    ┌──┴──┐            grounded)
          │         rerank)   │accept│
          │                   │refine├──► Re-retrieve
          │                   │web   ├──► PubMed live
          │                   └──────┘
```

- **Hybrid Retrieval**: BGE-M3 (1024d) dense + BM25 sparse, fused via RRF, reranked with cross-encoder
- **CRAG**: Corrective RAG with accept/refine/web_search decisions
- **Generation**: Chain-of-Thought grounding protocol (Extract → Verify → Compose)
- **Corpus**: 3,040 chunks from PubMed, CrossRef, PeruNPDB, GBIF, WFO (8 Peruvian species)

## Evaluation Results (DeepSeek backend)

| Metric | Score |
|--------|-------|
| BERTScore F1 (roberta-large) | 0.9028 |
| Semantic Similarity | 0.9929 |
| Context Precision | 1.0000 |
| Context Recall | 1.0000 |
| MRR | 1.0000 |
| NDCG@5 | 1.0000 |
| Entity Recall | 0.7946 |
| Faithfulness | 0.8411 |
| Answer Relevancy | 0.9995 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation benchmark
python -m evaluation.benchmark --backend deepseek

# Start web service
python -m web.app
# Open http://localhost:8000
```

## Docker

```bash
# Set your API key
export DEEPSEEK_API_KEY=your-key-here

# Build and run
docker compose up --build
```

## Target Species

1. *Uncaria tomentosa* (Cat's Claw)
2. *Lepidium meyenii* (Maca)
3. *Croton lechleri* (Dragon's Blood)
4. *Minthostachys mollis* (Muna)
5. *Erythroxylum coca* (Coca)
6. *Smallanthus sonchifolius* (Yacon)
7. *Physalis peruviana* (Aguaymanto)
8. *Buddleja incana* (Quishuar)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Interactive frontend |
| GET | `/api/health` | System status |
| POST | `/api/query` | Full pipeline query |
| GET | `/api/species` | Target species list |

## Backends

- **DeepSeek V4 Flash** — Best quality (API, requires key)
- **Ollama (Qwen3.5)** — Local inference (requires Ollama running)
- **Template** — Fast test mode (context passthrough)
