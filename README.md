# SIRCA-RAG

**Hybrid Corrective RAG for grounded question answering on Peruvian medicinal plants**

> Support repository for the paper *"Corrective RAG Híbrido para Generación Fundamentada sobre Plantas Medicinales Peruanas"* (SimBig / WAIMLAp 2026 / ACSAR). Contains the pipeline, the open corpus tooling, and the full reviewer-response experiment suite with reproducible results.

SIRCA-RAG (System for Intelligent Retrieval and Corrective Answers) answers questions about Peruvian medicinal plants using scientific literature. It couples **dense** (`multilingual-e5-base`) and **sparse** (BM25) retrieval via Reciprocal Rank Fusion, reranks with a cross-encoder, routes low-confidence retrievals through a **calibrated Corrective-RAG** stage, and generates answers under a three-step Chain-of-Thought protocol (Extract → Verify → Compose) that forces inline DOI/PMID citation.

---

## Headline results

Full pipeline on the 50-query bilingual benchmark (DeepSeek V4-Flash generator):

![Headline metrics](docs/images/headline_metrics.png)

| Metric | Score | Note |
|---|---|---|
| Context Recall@10 | **0.548** | not compared numerically against prior work — the closest related Spanish-language system (Collanqui et al.) evaluates a single-document, k=1-retrieval setup too different from ours for a valid comparison |
| MRR | **0.862** | most relevant doc at rank 1–2 on the large majority of queries |
| NDCG@10 | **0.821** | |
| BERTScore F1 (`roberta-large`) | **0.840** | |
| Fidelity (hybrid 65% semantic / 35% lexical) | **0.554** | conservative by design to suppress pharmacological drift |

---

## What the experiments actually show

### 1. The cross-encoder reranker directionally helps Fidelity — but not to a statistically robust degree

Five-configuration ablation. Removing the reranker causes a Fidelity drop, and `full` beat `no_reranker` in **every one of 5 independent re-runs** of this specific test performed during this work (0.534>0.467, 0.566>0.465, 0.517>0.469, 0.562>0.514, 0.554>0.526 — see "Notes on reproducibility" below for the full history). The direction is consistent; **formal statistical significance is not** — the paired Wilcoxon signed-rank p-value ranged from 0.00023 to 0.160 across those 5 runs, including two runs at n=80 giving p=0.028 and p=0.160. The final run, on the fully bug-fixed pipeline, gives **−5.0% relative (0.554→0.526), p=0.160 two-sided / p=0.080 one-sided — not significant at α=0.05**. We report this as a directionally consistent but not robustly significant finding rather than picking the most favorable run. At the retrieval-metric level, all five configurations are statistically equivalent to `full` (no p-value below 0.05) — `dense_only` is exactly invariant to the reranker (0 differing queries out of 50), confirmed as genuine (not a benchmark artifact) via a held-out α sweep that also led us to remove the query classifier's per-category retrieval weighting: it did not beat a flat α even when properly tuned on held-out data.

![Ablation Fidelity](docs/images/ablation_fidelity.png)

### 2. The corrective branches work — after a calibration fix

The default within-batch (min-max) score normalization makes the CRAG threshold relative, so *everything* is accepted. Replacing it with an **absolute sigmoid calibration** (accept ≥ 0.60, refine ≥ 0.30) makes the corrective branches fire as designed. On a 27-query out-of-distribution stress test, **off-domain queries route to web search 10/10**, and catalogued-but-unindexed species route to refine/web-search 8/9.

![CRAG routing](docs/images/crag_routing.png)

### 3. Robust to the choice of generator LLM

Re-running the full pipeline with a second generator (Cerebras Gemma-4-31B) on the same 50 queries: Gemma wins only on **Fidelity** (paired t-test, p = 0.0076) — its more literal, verbatim-preserving style specifically helps the lexical-overlap component. DeepSeek wins on BERTScore F1 (p = 0.0005); Semantic Similarity, Entity Recall, and Answer Relevancy show no significant difference (the latter borderline at p = 0.054). The pipeline's conclusions are not an artifact of one model, but the choice of generator does matter for Fidelity specifically.

![Cross-LLM](docs/images/cross_llm.png)

### 4. Retrieval generalizes beyond the evaluation slice

Extending the retrieval evaluation from 12 human-verified species to 30 additional stratified species (42 unique species total) holds up — metrics on the new species are at least as good as on the original 12 (MRR 0.866→0.883, Context Recall 0.554→0.550) — the retriever is not overfit to the benchmark subset.

![Coverage generalization](docs/images/coverage_generalization.png)

---

## Data corpus

- **100 medicinal plant species** from five southern Andean regions (Arequipa, Cusco, Puno, Moquegua, Tacna).
- **16,486 unique articles** after DOI deduplication, from **8 sources**: PubMed, Europe PMC, Semantic Scholar, CrossRef, GBIF, PeruNPDB, WFO, COCONUT.
- **6,098 indexed chunks** (512 tokens, 64 overlap, capped 100/species). **91/100** species contributed at least one chunk; the 9 without indexed literature are documented and used as a CRAG stress-test family.
- Embeddings: `intfloat/multilingual-e5-base` (768-dim). Reranker: `cross-encoder/ms-marco-MiniLM-L-12-v2`.

---

## Reproducing the experiments

```bash
pip install -r requirements.txt
# API keys are read from the environment (never hardcoded):
export DEEPSEEK_API_KEY=...      # DeepSeek generator (model id: deepseek-v4-flash)
export CEREBRAS_API_KEY=...      # optional: cross-LLM comparison (Gemma-4-31B)
```

| Script | Produces | Reviewer item |
|---|---|---|
| `run_perquery_agent_ablation.py` | per-query ablation + Wilcoxon (`results/wilcoxon_agent_vs_full.json`) | O4, O5 |
| `run_n5_fidelity_wilcoxon.py` | Fidelity significance, full vs no_reranker (`results/n5_fidelity_wilcoxon.json`) | N5 |
| `run_crag_stress_absolute.py` | CRAG routing validation (`results/crag_stress_absolute.json`) | O1 |
| `run_table2_extended.py` | coverage generalization (42 species) | O2 |
| `run_alpha_sweep_heldout.py` | held-out validation of classifier alpha weighting (`results/alpha_sweep_heldout.json`) | design soundness |
| `run_multi_llm_bench.py` | DeepSeek vs Cerebras comparison (`results/multi_llm_*.json`) | O9 |
| `run_llm_judge.py` | LLM-as-judge fidelity validation (`results/llm_judge_*.json`) | O7 |
| `docs/make_figures.py` | regenerates the figures in `docs/images/` | — |

All numeric results live in `results/*.json` and are the source of the figures above.

---

## Pipeline commands

```bash
python pipeline.py status      # corpus + vectorstore stats
python pipeline.py vectorize   # rebuild FAISS + BM25 indexes
python pipeline.py serve       # FastAPI service at http://localhost:8000
```

Live demo: <https://rag.scn.quest> — inspect retrieved DOIs/PMIDs and the CRAG routing decision per query in real time.

---

## Repository layout

```
agent/        CRAG agent (graph, CRAG evaluator, query classifier)
retrieval/    hybrid retriever (FAISS + BM25 + cross-encoder)
generation/   grounded generator (DeepSeek / Ollama / template backends)
ingestion/    source clients (PubMed, EPMC, S2, CrossRef, GBIF, PeruNPDB, WFO, COCONUT)
evaluation/   metrics, benchmark set, ablation harness
scraping/     CRAG web-search fallback
web/          FastAPI service + frontend
results/      experiment outputs (JSON) — figures are derived from these
docs/         figures and figure-generation script
run_*.py      reviewer-response experiment runners
```

---

## Notes on reproducibility

Generation-side metrics depend on a commercial API model (`deepseek-v4-flash`) that is not version-frozen; absolute generation scores can drift between runs. The **direction** of the findings reported here (reranker → Fidelity, cross-LLM comparison, CRAG routing) held stable across every re-run performed during this work — but **formal statistical significance did not**. Across the full debugging session, the reranker→Fidelity Wilcoxon test was run 5 times, each time on a codebase where at least one real bug had just been fixed:

| Run | full | no_reranker | Δ relative | n | p (two-sided) |
|---|---|---|---|---|---|
| 1 | 0.534 | 0.467 | −12.6% | 50 | 0.016 |
| 2 | 0.566 | 0.465 | −17.8% | 50 | 0.00023 |
| 3 | 0.517 | 0.469 | −9.2% | 50 | 0.110 |
| 4 | 0.562 | 0.514 | −8.5% | 80 | 0.028 |
| 5 (final, fully bug-fixed) | 0.554 | 0.526 | −5.0% | 80 | 0.160 |

Extending the sample from 50 to 80 queries did **not** stabilize the result — the same n=80 test gave both a significant (run 4) and a non-significant (run 5) p-value depending on which bugs were still present in the code (an `alpha`-restore-order bug in `agent/graph.py`, an unvalidated per-category classifier weighting, and a hash-randomization non-determinism bug in `agent/crag_evaluator.py::_refine_query()` — see below). We report this instability explicitly, using the final, fully-corrected run (5) as authoritative, rather than picking the run that looks best: the reranker's benefit to Fidelity should be read as a **consistent directional signal (5/5 runs), not a statistically confirmed effect**. Absolute generation values should be read as of the experiment date rather than as fixed constants.

Three evaluation-harness bugs were found and fixed during this work, all worth knowing about if you extend this code: (1) `evaluation/metrics.py`'s `faithfulness()` details key is `per_sample`, not `per_query` — a mismatch silently zeroed out the LLM-judge correlation in an earlier version of `run_llm_judge.py`; (2) HTTP clients here retry on HTTP-level errors (429/5xx) but transient DNS/socket failures raise `URLError`, which used to bypass retry entirely and silently corrupt aggregate metrics (19/50 blank answers in one run) — both `run_multi_llm_bench.py` and `run_llm_judge.py` now retry on `URLError` too; (3) `agent/crag_evaluator.py::_refine_query()` truncated a `set()` of expansion terms with `list(set())[:4]` — Python randomizes string hashing per process, so which terms survived the truncation (and thus the refined query fed into corrective re-retrieval) varied across separate runs. Fixed with `sorted(set())[:4]`. This is very likely a real contributor to the p-value instability documented above.
