# SIRCA-RAG

**Semi-Autonomous RAG for Peruvian Medicinal Plant Knowledge Integration**

> SimBig / WAIMLAp 2026

---

## Architecture

```mermaid
graph LR
    Q[Query] --> C[Classify]
    C --> R[Hybrid Retrieve]
    R --> E[CRAG Evaluate]
    E -- "accept >= 0.35" --> G[Generate]
    E -- "refine 0.15-0.35" --> RF[Re-retrieve]
    E -- "web < 0.15" --> WS[PubMed Live]
    RF --> G
    WS --> G
    G --> A[Grounded Answer]

    style Q fill:#3498db,stroke:#2c3e50,color:#fff
    style C fill:#9b59b6,stroke:#2c3e50,color:#fff
    style R fill:#e67e22,stroke:#2c3e50,color:#fff
    style E fill:#e74c3c,stroke:#2c3e50,color:#fff
    style G fill:#27ae60,stroke:#2c3e50,color:#fff
    style A fill:#2ecc71,stroke:#2c3e50,color:#fff
    style RF fill:#f39c12,stroke:#2c3e50,color:#fff
    style WS fill:#f39c12,stroke:#2c3e50,color:#fff
```

### Pipeline Stages

```mermaid
graph TD
    subgraph Classification
        CL[Rule-based Classifier]
        CL --> F[factual]
        CL --> EX[exploratory]
        CL --> CO[comparative]
    end

    subgraph Retrieval
        BGE[BGE-M3 Dense 1024d]
        BM[BM25 Sparse]
        BGE --> RRF[RRF Fusion]
        BM --> RRF
        RRF --> RK[Cross-Encoder Rerank]
    end

    subgraph CRAG
        TH[Threshold Decision]
        TH -- ">= 0.35" --> ACC[Accept]
        TH -- "0.15 - 0.35" --> REF[Refine]
        TH -- "< 0.15" --> WEB[Web Search]
    end

    subgraph Generation
        COT[Chain-of-Thought]
        COT --> S1[Extract facts]
        S1 --> S2[Verify sources]
        S2 --> S3[Compose answer]
    end

    style BGE fill:#3498db,stroke:#2c3e50,color:#fff
    style BM fill:#3498db,stroke:#2c3e50,color:#fff
    style RRF fill:#e67e22,stroke:#2c3e50,color:#fff
    style RK fill:#e67e22,stroke:#2c3e50,color:#fff
    style ACC fill:#27ae60,stroke:#2c3e50,color:#fff
    style REF fill:#f39c12,stroke:#2c3e50,color:#fff
    style WEB fill:#e74c3c,stroke:#2c3e50,color:#fff
    style S3 fill:#27ae60,stroke:#2c3e50,color:#fff
```

### Pipeline Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant C as Classifier
    participant R as Retriever
    participant E as CRAG Evaluator
    participant G as Generator

    U->>C: Query
    C->>C: Regex classify
    C->>R: category + weights
    R->>R: BGE-M3 encode
    R->>R: BM25 score
    R->>R: RRF fusion
    R->>R: Cross-encoder rerank
    R->>E: Top-5 documents + scores
    E->>E: Threshold check
    alt accept
        E->>G: Documents
    else refine
        E->>R: Re-retrieve
        R->>G: New documents
    else web_search
        E->>G: PubMed live results
    end
    G->>G: Extract - Verify - Compose
    G->>U: Grounded answer + citations
```

---

## Data Corpus

| # | Species | Common Name | Key Compounds |
|---|---------|-------------|---------------|
| 1 | *Uncaria tomentosa* | Cat's Claw / Una de Gato | Alkaloids, oxindoles |
| 2 | *Lepidium meyenii* | Maca | Macamides, glucosinolates |
| 3 | *Croton lechleri* | Dragon's Blood / Sangre de Grado | Taspine, proanthocyanidins |
| 4 | *Minthostachys mollis* | Muna | Pulegone, menthone |
| 5 | *Erythroxylum coca* | Coca | Cocaine alkaloids, flavonoids |
| 6 | *Smallanthus sonchifolius* | Yacon | FOS, phenolic acids |
| 7 | *Physalis peruviana* | Aguaymanto / Cape Gooseberry | Withanolides, carotenoids |
| 8 | *Buddleja incana* | Quishuar / Kiswar | Flavonoids, iridoids |

- **3,040** vectorized chunks from **6 sources**: PubMed, CrossRef, PeruNPDB, GBIF, WFO, SciELO
- **Embedding**: BAAI/bge-m3 (1024 dimensions)
- **Chunking**: 512 tokens, 64 overlap

---

## Evaluation Results (DeepSeek Backend)

| Metric | Score | Target | Status |
|--------|------:|--------|--------|
| BERTScore F1 (roberta-large) | **0.9028** | >= 0.90 | PASS |
| Semantic Similarity (cross-encoder) | **0.9929** | -- | -- |
| Context Precision | **1.0000** | -- | -- |
| Context Recall | **1.0000** | -- | -- |
| MRR | **1.0000** | -- | -- |
| NDCG@5 | **1.0000** | -- | -- |
| Entity Recall | **0.7946** | -- | -- |
| Faithfulness | **0.8411** | >= 0.80 | PASS |
| Answer Relevancy | **0.9995** | -- | -- |

### Faithfulness Metric Composition

```mermaid
pie title Hybrid Faithfulness Score
    "Semantic - cross-encoder" : 65
    "Lexical - word overlap" : 35
```

---

## Pipeline Trace Example

```
Query: "What are the main alkaloids in Uncaria tomentosa?"
```

```mermaid
gantt
    title Pipeline Execution Timeline
    dateFormat x
    axisFormat %L ms

    section Classify
    Rule-based classify :0, 4

    section Retrieve
    BGE-M3 + BM25 + Rerank :4, 3199

    section Evaluate
    Threshold check :3199, 3200

    section Generate
    DeepSeek V4 Flash :3200, 6200
```

| Node | Duration | Details |
|------|----------|---------|
| classify | < 1ms | category: exploratory, confidence: 0.60 |
| retrieve | 3,195ms | 5 docs, hybrid alpha=0.6, BGE-M3 + BM25 + rerank |
| evaluate | < 1ms | action: accept, confidence: 0.89 |
| generate | ~30-150s | DeepSeek: API call / Template: < 1ms / Ollama: ~10-30s |

---

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

---

## Docker Deployment (Dokploy)

```mermaid
graph LR
    subgraph Docker
        APP[FastAPI :8000]
        VS[FAISS 3040 vectors]
        BGE[BGE-M3]
        APP --> VS
        APP --> BGE
    end

    subgraph External
        DS[DeepSeek API]
        OL[Ollama optional]
    end

    APP -- DEEPSEEK_API_KEY --> DS
    APP -. OLLAMA_BASE_URL .-> OL
    U[User] -- ":8000" --> APP

    style APP fill:#27ae60,stroke:#2c3e50,color:#fff
    style DS fill:#3498db,stroke:#2c3e50,color:#fff
    style OL fill:#95a5a6,stroke:#2c3e50,color:#fff
    style U fill:#9b59b6,stroke:#2c3e50,color:#fff
```

```bash
# Set your API key
export DEEPSEEK_API_KEY=your-key-here

# Build and run
docker compose up --build
```

| Variable | Required | Default |
|----------|----------|---------|
| `DEEPSEEK_API_KEY` | Yes | -- |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` |
| `HOST` | No | `0.0.0.0` |
| `PORT` | No | `8000` |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive dark-themed frontend |
| `GET` | `/api/health` | System status, backends, vectorstore size |
| `POST` | `/api/query` | Full pipeline query with answer + citations + trace |
| `GET` | `/api/species` | List of 8 target species |

### Query Request / Response

```mermaid
classDiagram
    class QueryRequest {
        +string query
        +string backend
        +string language
    }

    class QueryResponse {
        +string query
        +string answer
        +list citations
        +dict classification
        +string crag_action
        +float confidence
        +string model
        +int tokens_generated
        +int latency_ms
        +list trace_summary
    }

    class Citation {
        +int index
        +string title
        +list authors
        +string year
        +string source
        +string pmid
        +string doi
        +list species
    }

    QueryRequest --> QueryResponse : POST /api/query
    QueryResponse --> Citation : contains
```

---

## Backends

| Backend | Type | Speed | Quality | Requirements |
|---------|------|-------|---------|--------------|
| **DeepSeek V4 Flash** | API | ~30-150s | Best | `DEEPSEEK_API_KEY` |
| **Ollama (Qwen3.5)** | Local | ~10-30s | Good | Ollama running |
| **Template** | Passthrough | < 1ms | Test only | None |

---

## Project Structure

```mermaid
graph TD
    subgraph Core
        AG[agent] --> GR[graph.py]
        AG --> QC[query_classifier.py]
        AG --> CE[crag_evaluator.py]
    end

    subgraph Retrieval
        RT[retrieval] --> HY[hybrid.py]
        RT --> BM[bm25_index.py]
    end

    subgraph Generation
        GN[generation] --> GG[grounded_generator.py]
    end

    subgraph Web
        WB[web] --> AP[app.py]
        WB --> ST[static]
    end

    subgraph Data
        DT[data] --> VEC[vectorstore]
        DT --> RAW[raw]
        DT --> PRO[processed]
    end

    style AG fill:#9b59b6,stroke:#2c3e50,color:#fff
    style RT fill:#e67e22,stroke:#2c3e50,color:#fff
    style GN fill:#27ae60,stroke:#2c3e50,color:#fff
    style WB fill:#3498db,stroke:#2c3e50,color:#fff
    style DT fill:#95a5a6,stroke:#2c3e50,color:#fff
```

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Chain-of-Thought Grounding | 3-step protocol (Extract, Verify, Compose) prevents hallucination |
| Hybrid Faithfulness | 65% semantic + 35% lexical catches paraphrases and exact matches |
| roberta-large for BERTScore | DeBERTa caused OverflowError; roberta-large gives stable F1 >= 0.90 |
| Temperature 0.0 | Deterministic output for reproducible evaluation |
| Bilingual (ES/EN) | System prompt auto-detects and responds in query language |
| Pre-computed rerank scores | Sub-millisecond CRAG evaluation without model inference |

---

*Built for SimBig / WAIMLAp 2026*
