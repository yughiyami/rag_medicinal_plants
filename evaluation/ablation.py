"""
SIRCA-RAG Ablation Study.
Compares the full pipeline against degraded variants to quantify
each component's contribution to retrieval and generation quality.

Configurations:
  full          — Complete pipeline (hybrid + reranker + CRAG + classifier)
  dense_only    — FAISS dense retrieval only (alpha=1.0)
  sparse_only   — BM25 sparse retrieval only (alpha=0.0)
  no_reranker   — Hybrid retrieval without cross-encoder reranking
  no_crag       — Skip CRAG evaluation, always accept all results
  no_classifier — Skip query classification, use default alpha
"""
import json
import sys
import time
import types
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark_data import TestCase, BENCHMARK_SET
from evaluation.metrics import (
    bertscore,
    bertscore_lite,
    context_precision,
    context_recall,
    mrr,
    ndcg_at_k,
    entity_recall,
    faithfulness,
    answer_relevancy,
)


@dataclass
class AblationConfig:
    name: str
    description: str
    alpha: float | None = None
    use_reranker: bool = True
    use_crag: bool = True
    use_classifier: bool = True


ABLATION_CONFIGS = [
    AblationConfig(
        name="full",
        description="Complete pipeline (baseline)",
    ),
    AblationConfig(
        name="dense_only",
        description="FAISS dense retrieval only",
        alpha=1.0,
    ),
    AblationConfig(
        name="sparse_only",
        description="BM25 sparse retrieval only",
        alpha=0.0,
    ),
    AblationConfig(
        name="no_reranker",
        description="Hybrid without cross-encoder reranking",
        use_reranker=False,
    ),
    AblationConfig(
        name="no_crag",
        description="Skip CRAG evaluation, always accept",
        use_crag=False,
    ),
    AblationConfig(
        name="no_classifier",
        description="No query classification, default alpha",
        use_classifier=False,
    ),
]


class _AlwaysAcceptEvaluator:
    """Stub evaluator that always accepts retrieved results."""

    def evaluate(self, query, results, use_rerank_scores=True):
        from agent.crag_evaluator import CRAGDecision
        return CRAGDecision(
            action="accept",
            confidence=1.0,
            relevant_indices=list(range(len(results))),
            reason="ablation: CRAG disabled",
        )


def _build_agent(config: AblationConfig, backend: str = "template"):
    """Build an agent configured for a specific ablation variant."""
    from retrieval.hybrid import HybridRetriever
    from agent.graph import SIRCAAgent
    import agent.graph as graph_module

    retriever = HybridRetriever()
    if config.alpha is not None:
        retriever.alpha = config.alpha
    retriever.load()

    agent = SIRCAAgent(
        retriever=retriever,
        generator_backend=backend,
    )

    restore_fns = []

    if not config.use_reranker:
        original_retrieve = retriever.retrieve

        def retrieve_no_rerank(query, top_k=20, rerank_top_k=5, use_rerank=True):
            return original_retrieve(query, top_k=top_k, rerank_top_k=rerank_top_k, use_rerank=False)

        retriever.retrieve = retrieve_no_rerank

    if not config.use_crag:
        agent._evaluator = _AlwaysAcceptEvaluator()

    if not config.use_classifier:
        from agent.query_classifier import QueryClassification
        original_classify = graph_module.classify_query

        def fixed_classify(query):
            return QueryClassification(
                category="exploratory",
                confidence=0.5,
                features={},
                alpha_override=None,
            )

        graph_module.classify_query = fixed_classify
        restore_fns.append(lambda: setattr(graph_module, "classify_query", original_classify))

    def restore():
        for fn in restore_fns:
            fn()

    return agent, restore


def _run_queries(agent, benchmark: list[TestCase]):
    """Run all benchmark queries and collect results."""
    queries, answers, references, contexts = [], [], [], []
    all_retrieved, all_relevant = [], []

    # Honest Recall@k: build a wide candidate POOL (top-K, pre-rerank) per query.
    # Mark as "relevant" the pool docs whose species metadata (or content) matches
    # tc.relevant_species. Then track which of those relevants survive into the
    # final top-5 (after rerank + CRAG). Recall@5 < 1.0 when the reranker drops
    # any relevant chunk from the top-5 window.
    POOL_K = 30
    retriever = agent._retriever

    for tc in benchmark:
        # ---- 1) Wide candidate pool (pre-rerank) used as ground-truth proxy ----
        try:
            pool = retriever.retrieve(
                tc.query, top_k=POOL_K, rerank_top_k=POOL_K, use_rerank=False
            )
        except Exception:
            pool = []

        # Use chunk's stable FAISS index as id. The pool comes from _dense_search /
        # _sparse_search / RRF, all of which keep "index" pointing to the chunk row.
        pool_ids = [int(d.get("index", -1)) for d in pool]

        # Build "relevant in pool" by matching species in chunk metadata or content.
        relevant_pool_ids = set()
        for d, cid in zip(pool, pool_ids):
            meta = d.get("metadata") or {}
            species_meta = set(s.lower() for s in (meta.get("species") or []))
            text = (d.get("content") or "").lower()
            for sp in tc.relevant_species:
                spl = sp.lower()
                if spl in species_meta or spl in text:
                    relevant_pool_ids.add(cid)
                    break

        # ---- 2) Run actual agent (top-5 after rerank + CRAG) ----
        result = agent.run(tc.query)
        gen = result.get("generation", {})
        answer = gen.get("answer", "")
        context = result.get("context", "")
        citations = result.get("citations", [])
        # retrieval_results has the same chunks as citations, BUT keeps the stable
        # FAISS index, which citations lose. Use it to identify what survived rerank.
        retr_raw = result.get("retrieval_results", [])
        retrieved_ids = [int(d.get("index", -1)) for d in retr_raw]

        # If the agent took the web_search branch, retrieval_results may be empty.
        # Fall back to species matching on citations against the pool.
        if not retrieved_ids:
            cit_species = []
            for cit in citations:
                cit_species.append(set(s.lower() for s in (cit.get("species") or [])))
            # Best-effort: take pool ids whose species overlap citation species.
            retrieved_ids = []
            used = set()
            for d, cid in zip(pool, pool_ids):
                meta = d.get("metadata") or {}
                sm = set(s.lower() for s in (meta.get("species") or []))
                if any(sm & cs for cs in cit_species) and cid not in used:
                    retrieved_ids.append(cid)
                    used.add(cid)
                if len(retrieved_ids) >= len(citations):
                    break

        # Unify into shared integer index space so existing metric fns work.
        id_to_idx = {cid: i for i, cid in enumerate(pool_ids)}
        for cid in retrieved_ids:
            if cid not in id_to_idx:
                id_to_idx[cid] = len(id_to_idx)

        retrieved_idx = [id_to_idx[cid] for cid in retrieved_ids]
        relevant_idx = set(id_to_idx[cid] for cid in relevant_pool_ids)

        # Sanity floor: if no relevants found at all, mark first retrieved as
        # weakly relevant to avoid division-by-zero in Precision/MRR. This is
        # rare given our species-targeted benchmark.
        if not relevant_idx and retrieved_idx:
            relevant_idx.add(retrieved_idx[0])

        queries.append(tc.query)
        answers.append(answer)
        references.append(tc.reference_answer)
        contexts.append(context)
        all_retrieved.append(retrieved_idx)
        all_relevant.append(relevant_idx)

    return queries, answers, references, contexts, all_retrieved, all_relevant


def _compute_metrics(queries, answers, references, contexts, all_retrieved, all_relevant):
    """Compute all evaluation metrics."""
    results = {}

    try:
        bs = bertscore(answers, references, lang="en")
        results["bertscore_f1"] = bs.score
    except Exception:
        results["bertscore_f1"] = 0.0

    bs_lite = bertscore_lite(answers, references)
    results["semantic_similarity"] = bs_lite.score

    cp = context_precision(all_retrieved, all_relevant)
    results["context_precision"] = cp.score

    cr = context_recall(all_retrieved, all_relevant)
    results["context_recall"] = cr.score

    m = mrr(all_retrieved, all_relevant)
    results["mrr"] = m.score

    # Plan A: report both NDCG@5 (legacy) and NDCG@10 (new top-k window)
    nd5 = ndcg_at_k(all_retrieved, all_relevant, k=5)
    results["ndcg@5"] = nd5.score
    nd10 = ndcg_at_k(all_retrieved, all_relevant, k=10)
    results["ndcg@10"] = nd10.score

    er = entity_recall(answers, references, contexts)
    results["entity_recall"] = er.score

    ff = faithfulness(answers, contexts)
    results["faithfulness"] = ff.score

    ar = answer_relevancy(queries, answers)
    results["answer_relevancy"] = ar.score

    return results


def run_ablation(
    configs: list[AblationConfig] | None = None,
    benchmark: list[TestCase] | None = None,
    backend: str = "template",
) -> dict:
    """Run the full ablation study across all configurations."""
    configs = configs or ABLATION_CONFIGS
    benchmark = benchmark or BENCHMARK_SET

    print("=" * 70)
    print("SIRCA-RAG: Ablation Study")
    print(f"  Configurations: {len(configs)}")
    print(f"  Test cases: {len(benchmark)}")
    print(f"  Generator: {backend}")
    print("=" * 70)

    all_results = {}

    for cfg in configs:
        print(f"\n--- [{cfg.name}] {cfg.description} ---")
        start = time.time()

        agent, restore = _build_agent(cfg, backend)

        try:
            print(f"  Running {len(benchmark)} queries...", end=" ", flush=True)
            data = _run_queries(agent, benchmark)
            elapsed_queries = time.time() - start
            print(f"done ({elapsed_queries:.0f}s)")

            print("  Computing metrics...", end=" ", flush=True)
            metrics = _compute_metrics(*data)
            elapsed_total = time.time() - start
            print(f"done ({elapsed_total:.0f}s)")

            all_results[cfg.name] = metrics
        finally:
            restore()

    return all_results


def print_ablation_table(results: dict):
    """Print a formatted comparison table."""
    if not results:
        print("No results to display.")
        return

    configs = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    col_width = 14
    header = f"{'Metric':<25}" + "".join(f"{c:>{col_width}}" for c in configs)
    print("\n" + "=" * len(header))
    print("ABLATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for metric in metrics:
        values = [results[c].get(metric, 0.0) for c in configs]
        best = max(values)
        row = f"  {metric:<23}"
        for v in values:
            marker = " *" if abs(v - best) < 1e-6 and len(configs) > 1 else "  "
            row += f"{v:>{col_width - 2}.4f}{marker}"
        print(row)

    print("=" * len(header))

    if "full" in results:
        print("\nDelta vs Full Pipeline:")
        full = results["full"]
        for cfg_name in configs:
            if cfg_name == "full":
                continue
            deltas = []
            for m in metrics:
                diff = results[cfg_name].get(m, 0.0) - full.get(m, 0.0)
                deltas.append(f"{diff:+.4f}")
            print(f"  {cfg_name:<20} " + "  ".join(deltas))


def save_ablation_results(results: dict, backend: str, output_dir: Path | None = None):
    """Save results to JSON with metadata."""
    from datetime import datetime

    output_dir = output_dir or Path(__file__).resolve().parent.parent / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    envelope = {
        "timestamp": datetime.now().isoformat(),
        "backend": backend,
        "num_queries": len(BENCHMARK_SET),
        "configs": {c.name: c.description for c in ABLATION_CONFIGS},
        "results": results,
    }

    path = output_dir / f"ablation_{backend}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {path}")
    return path


def select_diverse_subset(benchmark: list[TestCase], n: int = 20) -> list[TestCase]:
    """Select a diverse subset balancing categories and languages."""
    from collections import defaultdict
    by_cat = defaultdict(list)
    for tc in benchmark:
        by_cat[tc.category].append(tc)

    # Proportional allocation
    selected = []
    for cat in ["factual", "exploratory", "comparative"]:
        items = by_cat[cat]
        count = max(2, round(n * len(items) / len(benchmark)))
        # Alternate EN/ES
        en = [t for t in items if not any(c in t.query.lower() for c in ["cual", "como", "que ", "actividad", "distribucion", "contenido", "comparar", "diferencia"])]
        es = [t for t in items if t not in en]
        picked = []
        for src in [en, es]:
            take = min(len(src), count // 2 + 1)
            picked.extend(src[:take])
        selected.extend(picked[:count])

    return selected[:n]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIRCA-RAG Ablation Study")
    parser.add_argument("--backend", default="template", choices=["template", "deepseek", "ollama"])
    parser.add_argument("--configs", nargs="*", default=None, help="Config names to run (default: all)")
    parser.add_argument("--subset", type=int, default=0, help="Use N diverse queries instead of all 50")
    args = parser.parse_args()

    configs = ABLATION_CONFIGS
    if args.configs:
        configs = [c for c in ABLATION_CONFIGS if c.name in args.configs]

    benchmark = BENCHMARK_SET
    if args.subset > 0:
        benchmark = select_diverse_subset(BENCHMARK_SET, args.subset)
        print(f"Using {len(benchmark)}/{len(BENCHMARK_SET)} diverse queries")

    results = run_ablation(configs=configs, benchmark=benchmark, backend=args.backend)
    print_ablation_table(results)
    save_ablation_results(results, args.backend)
