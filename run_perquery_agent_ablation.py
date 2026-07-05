"""
Per-query ablation via the REAL agent pipeline (reviewer N1 fix).

Uses evaluation/ablation.py's _build_agent + _run_queries so the per-query
retrieval metrics come from the SAME methodology that produced Table 4
(agent-based, rerank_top_k=5, CRAG-aware). This guarantees Table 5 (Wilcoxon)
is internally consistent with Table 4 and that dense_only / no_classifier
are measured as genuinely distinct configurations.

Template backend => no LLM API calls; retrieval metrics are backend-independent.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.ablation import ABLATION_CONFIGS, _build_agent, _run_queries

# Stub the web searcher so the CRAG web_search branch neither performs a live
# network call nor crashes on asyncio cleanup. retrieval_results is captured in
# the retrieve node BEFORE routing, so this does not alter retrieval metrics; it
# only prevents the corrective branch from doing real work during measurement.
import scraping.web_searcher as _ws


class _StubSearcher:
    def __init__(self, *a, **k):
        pass

    async def search(self, *a, **k):
        return []

    async def close(self):
        return None


_ws.WebSearcher = _StubSearcher
from evaluation.benchmark_data import BENCHMARK_SET
from evaluation.metrics import context_precision, context_recall, mrr, ndcg_at_k

from scipy.stats import wilcoxon


def main():
    print("=" * 70)
    print("Per-query AGENT-based ablation (matches Table 4 methodology)")
    print("=" * 70)

    per_query = {}
    aggregate = {}

    for cfg in ABLATION_CONFIGS:
        print(f"\n--- {cfg.name} ---")
        t0 = time.time()
        agent, restore = _build_agent(cfg, backend="template")
        # Real CRAG evaluator retained (reproduces Table 4). The web searcher is
        # globally stubbed above so any web_search routing is a harmless no-op.
        queries, answers, references, contexts, all_retrieved, all_relevant = _run_queries(
            agent, BENCHMARK_SET
        )
        cp = context_precision(all_retrieved, all_relevant)
        cr = context_recall(all_retrieved, all_relevant)
        m = mrr(all_retrieved, all_relevant)
        nd10 = ndcg_at_k(all_retrieved, all_relevant, k=10)
        per_query[cfg.name] = {
            "context_precision": cp.details["per_query"],
            "context_recall": cr.details["per_query"],
            "mrr": m.details["per_query"],
            "ndcg@10": nd10.details["per_query"],
        }
        aggregate[cfg.name] = {
            "context_precision": cp.score,
            "context_recall": cr.score,
            "mrr": m.score,
            "ndcg@10": nd10.score,
        }
        print(f"  C.Prec={cp.score:.3f} C.Recall={cr.score:.3f} "
              f"MRR={m.score:.3f} NDCG@10={nd10.score:.3f} "
              f"({time.time()-t0:.1f}s)")
        restore()

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "per_query_agent_ablation.json").write_text(
        json.dumps({"per_query": per_query, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("Wilcoxon signed-rank vs full (agent-based per-query)")
    print("=" * 70)
    print(f"{'config':16s} | {'metric':18s} | {'W':>8s} {'p':>10s} {'n_nz':>5s}")

    full = per_query["full"]
    wil = {}
    for name in per_query:
        if name == "full":
            continue
        wil[name] = {}
        for metric in ("context_precision", "context_recall", "mrr", "ndcg@10"):
            a, b = full[metric], per_query[name][metric]
            diffs = [bi - ai for ai, bi in zip(a, b)]
            nz = [d for d in diffs if d != 0]
            if not nz:
                stat, p, note = float("nan"), 1.0, "all-zero diffs"
            else:
                try:
                    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                    note = ""
                except ValueError as e:
                    stat, p, note = float("nan"), float("nan"), str(e)
            wil[name][metric] = {
                "statistic": None if stat != stat else float(stat),
                "p_value": None if p != p else float(p),
                "n_nonzero_diffs": len(nz),
                "note": note,
            }
            wd = f"{stat:8.2f}" if stat == stat else "     nan"
            pd = f"{p:10.4f}" if p == p else "       nan"
            print(f"{name:16s} | {metric:18s} | {wd} {pd} {len(nz):5d}")

    (out / "wilcoxon_agent_vs_full.json").write_text(json.dumps(wil, indent=2), encoding="utf-8")
    print("\nSaved -> results/per_query_agent_ablation.json + wilcoxon_agent_vs_full.json")


if __name__ == "__main__":
    main()
