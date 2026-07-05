"""
Per-query retrieval ablation for statistical testing (reviewer O4, O5).

Runs six retrieval-only ablation variants directly on the HybridRetriever
(bypassing the LangGraph agent so we can measure retrieval quality without
touching the LLM). Saves per-query Recall/MRR/NDCG/Precision arrays and
runs Wilcoxon signed-rank tests vs the `full` configuration.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.benchmark_data import BENCHMARK_SET
from evaluation.metrics import (
    context_precision,
    context_recall,
    mrr,
    ndcg_at_k,
)
from retrieval.hybrid import HybridRetriever
from agent.query_classifier import classify_query

from scipy.stats import wilcoxon


# (name, alpha_mode, use_reranker)
# alpha_mode: "classifier" | "fixed:VALUE"
CONFIGS = [
    ("full",          "classifier",  True),
    ("dense_only",    "fixed:1.0",   True),
    ("sparse_only",   "fixed:0.0",   True),
    ("no_reranker",   "classifier",  False),
    ("no_crag",       "classifier",  True),   # CRAG only affects routing, not retrieval
    ("no_classifier", "fixed:0.6",   True),
]

POOL_K = 30
TOP_K = 10


def _resolve_alpha(mode: str, query: str) -> float | None:
    if mode == "classifier":
        cls = classify_query(query)
        return cls.alpha_override  # None if classifier didn't set one
    if mode.startswith("fixed:"):
        return float(mode.split(":", 1)[1])
    return None


def _run_retrieval(retriever: HybridRetriever, alpha_mode: str, use_rerank: bool,
                   default_alpha: float):
    all_retrieved, all_relevant = [], []

    for tc in BENCHMARK_SET:
        alpha = _resolve_alpha(alpha_mode, tc.query)
        retriever.alpha = alpha if alpha is not None else default_alpha
        try:
            pool = retriever.retrieve(
                tc.query, top_k=POOL_K, rerank_top_k=POOL_K, use_rerank=False
            )
        except Exception:
            pool = []
        pool_ids = [int(d.get("index", -1)) for d in pool]

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

        try:
            retrieved = retriever.retrieve(
                tc.query, top_k=POOL_K, rerank_top_k=TOP_K, use_rerank=use_rerank
            )
        except Exception:
            retrieved = []
        retrieved_ids = [int(d.get("index", -1)) for d in retrieved]

        id_to_idx = {cid: i for i, cid in enumerate(pool_ids)}
        for cid in retrieved_ids:
            if cid not in id_to_idx:
                id_to_idx[cid] = len(id_to_idx)

        retrieved_idx = [id_to_idx[cid] for cid in retrieved_ids]
        relevant_idx = set(id_to_idx[cid] for cid in relevant_pool_ids)

        if not relevant_idx and retrieved_idx:
            relevant_idx.add(retrieved_idx[0])

        all_retrieved.append(retrieved_idx)
        all_relevant.append(relevant_idx)

    return all_retrieved, all_relevant


def main():
    print("=" * 70)
    print("Per-query retrieval ablation + Wilcoxon (bypass LangGraph)")
    print(f"Configs: {len(CONFIGS)}  Queries: {len(BENCHMARK_SET)}")
    print("=" * 70)

    retriever = HybridRetriever()
    retriever.load()
    original_alpha = retriever.alpha

    per_query = {}

    for name, alpha_mode, use_rerank in CONFIGS:
        print(f"\n--- {name} (alpha_mode={alpha_mode}, rerank={use_rerank}) ---")
        t0 = time.time()
        all_retrieved, all_relevant = _run_retrieval(
            retriever, alpha_mode, use_rerank, default_alpha=original_alpha
        )

        cp = context_precision(all_retrieved, all_relevant)
        cr = context_recall(all_retrieved, all_relevant)
        m = mrr(all_retrieved, all_relevant)
        nd10 = ndcg_at_k(all_retrieved, all_relevant, k=10)

        per_query[name] = {
            "context_precision": cp.details["per_query"],
            "context_recall": cr.details["per_query"],
            "mrr": m.details["per_query"],
            "ndcg@10": nd10.details["per_query"],
        }
        print(
            f"  C.Prec={cp.score:.3f}  C.Recall={cr.score:.3f}  "
            f"MRR={m.score:.3f}  NDCG@10={nd10.score:.3f}  "
            f"elapsed={time.time()-t0:.1f}s"
        )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    pq_path = out_dir / "per_query_ablation_retrieval.json"
    pq_path.write_text(json.dumps(per_query, indent=2), encoding="utf-8")
    print(f"\nPer-query arrays saved -> {pq_path}")

    print("\n" + "=" * 70)
    print("Wilcoxon signed-rank vs full")
    print("=" * 70)
    print(f"{'config':16s} | {'metric':18s} | {'W':>8s} {'p_value':>10s} {'n_nz':>6s}")

    full = per_query["full"]
    wilcoxon_results = {}
    for name in per_query:
        if name == "full":
            continue
        wilcoxon_results[name] = {}
        for metric in ("context_precision", "context_recall", "mrr", "ndcg@10"):
            a = full[metric]
            b = per_query[name][metric]
            diffs = [bi - ai for ai, bi in zip(a, b)]
            non_zero = [d for d in diffs if d != 0]
            if not non_zero:
                stat, p, note = float("nan"), 1.0, "all-zero diffs"
            else:
                try:
                    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                    note = ""
                except ValueError as e:
                    stat, p, note = float("nan"), float("nan"), str(e)
            wilcoxon_results[name][metric] = {
                "statistic": None if stat != stat else float(stat),
                "p_value": None if p != p else float(p),
                "n_nonzero_diffs": len(non_zero),
                "note": note,
            }
            wdisp = f"{stat:8.2f}" if stat == stat else "     nan"
            pdisp = f"{p:10.4f}" if p == p else "       nan"
            print(f"{name:16s} | {metric:18s} | {wdisp} {pdisp} {len(non_zero):6d}")

    wpath = out_dir / "wilcoxon_vs_full.json"
    wpath.write_text(json.dumps(wilcoxon_results, indent=2), encoding="utf-8")
    print(f"\nWilcoxon results saved -> {wpath}")


if __name__ == "__main__":
    main()
