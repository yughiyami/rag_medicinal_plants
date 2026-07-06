"""
Held-out alpha sweep for the query classifier.

Motivation: the ablation (Table 4/5) showed no_classifier (flat alpha=0.6)
significantly beating full (classifier alpha 0.3/0.7/0.5) on retrieval. That
comparison used the same 50-query benchmark that the paper reports numbers
on, so it cannot answer whether a *properly tuned* alpha (flat or per-category)
would do better -- tuning and reporting on the same set would be overfitting
to the test set, exactly the kind of thing this project has been auditing
other people's numbers for.

Methodology: tune exclusively on the 30 held-out species from
run_table2_extended.py's build_new_species_cases() (never touched by Table 4/5
or the reported "full" config). Only the winning configuration(s) get evaluated
once on the actual BENCHMARK_SET (12 species, 50 queries) for a final,
un-overfit comparison against `full` (classifier 0.3/0.7/0.5) and `no_classifier`
(flat 0.6).

Retrieval-only (backend="template"): no LLM calls, no API cost.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scraping.web_searcher as _ws


class _StubSearcher:
    def __init__(self, *a, **k):
        pass

    async def search(self, *a, **k):
        return []

    async def close(self):
        return None


_ws.WebSearcher = _StubSearcher

import agent.graph as graph_module
from agent.query_classifier import QueryClassification, classify_query as real_classify_query
from evaluation.ablation import AblationConfig, _build_agent, _run_queries
from evaluation.benchmark_data import BENCHMARK_SET
from evaluation.metrics import context_precision, context_recall, mrr, ndcg_at_k
from run_table2_extended import build_new_species_cases

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def _measure(cases, alpha_override_fn):
    """Run the full agent (reranker + CRAG on) over `cases` with a custom
    classify_query() that assigns alpha via alpha_override_fn(category)."""
    cfg = AblationConfig(name="sweep", description="")
    agent, restore = _build_agent(cfg, backend="template")

    def patched_classify(query):
        base = real_classify_query(query)
        return QueryClassification(
            category=base.category,
            confidence=base.confidence,
            features=base.features,
            alpha_override=alpha_override_fn(base.category),
        )

    graph_module.classify_query = patched_classify
    try:
        q, a, r, c, retr, rel = _run_queries(agent, cases)
        return {
            "context_precision": context_precision(retr, rel).score,
            "context_recall": context_recall(retr, rel).score,
            "mrr": mrr(retr, rel).score,
            "ndcg@10": ndcg_at_k(retr, rel, k=10).score,
        }
    finally:
        graph_module.classify_query = real_classify_query
        restore()


def _composite(m):
    """Equal-weight composite of the four retrieval metrics for ranking alphas."""
    return (m["context_precision"] + m["context_recall"] + m["mrr"] + m["ndcg@10"]) / 4.0


def main():
    print("=" * 70)
    print("Held-out alpha sweep (tuning set: 30 new species, never the reported benchmark)")
    print("=" * 70)

    heldout = build_new_species_cases(30, seed=42)

    # Bucket held-out queries by their REAL classifier category (from the actual
    # query text, not the TestCase.category metadata field which is unused here).
    by_category = {"factual": [], "exploratory": [], "comparative": []}
    for tc in heldout:
        cat = real_classify_query(tc.query).category
        by_category[cat].append(tc)
    for cat, cases in by_category.items():
        print(f"  held-out queries classified as {cat}: {len(cases)}")

    # 1) Flat alpha sweep on the FULL held-out set.
    print("\n--- Flat alpha sweep (held-out, n=30) ---")
    flat_results = []
    for alpha in GRID:
        m = _measure(heldout, lambda _cat, a=alpha: a)
        m["alpha"] = alpha
        m["composite"] = _composite(m)
        flat_results.append(m)
        print(f"  alpha={alpha:.1f}  CP={m['context_precision']:.3f} CR={m['context_recall']:.3f} "
              f"MRR={m['mrr']:.3f} NDCG10={m['ndcg@10']:.3f}  composite={m['composite']:.3f}")
    best_flat = max(flat_results, key=lambda m: m["composite"])
    print(f"  BEST flat alpha on held-out: {best_flat['alpha']:.1f} (composite={best_flat['composite']:.3f})")

    # 2) Per-category alpha sweep: for each category, find its own best alpha
    #    using ONLY that category's held-out queries.
    print("\n--- Per-category alpha sweep (held-out) ---")
    best_per_cat = {}
    for cat, cases in by_category.items():
        if not cases:
            print(f"  {cat}: no held-out queries, skipping")
            continue
        cat_results = []
        for alpha in GRID:
            m = _measure(cases, lambda _c, a=alpha: a)
            m["alpha"] = alpha
            m["composite"] = _composite(m)
            cat_results.append(m)
        best = max(cat_results, key=lambda m: m["composite"])
        best_per_cat[cat] = best["alpha"]
        print(f"  {cat} (n={len(cases)}): best alpha={best['alpha']:.1f} "
              f"(composite={best['composite']:.3f} vs flat-0.6 on same subset "
              f"{next(r for r in cat_results if r['alpha']==0.6)['composite']:.3f})")

    # 3) Final, single evaluation on the REPORTED 50-query/12-species benchmark
    #    (touched only here, once) -- comparing: current classifier (0.3/0.7/0.5),
    #    flat 0.6 (=no_classifier from Table 4/5), best flat alpha from held-out,
    #    and tuned per-category alpha from held-out.
    print("\n" + "=" * 70)
    print("Final held-out-tuned comparison on the REPORTED benchmark (50 queries, touched once)")
    print("=" * 70)

    current_cfg = {"factual": 0.3, "exploratory": 0.7, "comparative": 0.5}
    configs_to_test = {
        "current_classifier (0.3/0.7/0.5)": lambda cat: current_cfg[cat],
        "flat_0.6 (=no_classifier)": lambda cat: 0.6,
        f"flat_best_heldout ({best_flat['alpha']:.1f})": lambda cat, a=best_flat["alpha"]: a,
        "per_category_heldout_tuned": lambda cat: best_per_cat.get(cat, 0.6),
    }

    final = {}
    for name, fn in configs_to_test.items():
        m = _measure(BENCHMARK_SET, fn)
        final[name] = m
        print(f"  {name:32s} CP={m['context_precision']:.3f} CR={m['context_recall']:.3f} "
              f"MRR={m['mrr']:.3f} NDCG10={m['ndcg@10']:.3f}")

    out = Path("results/alpha_sweep_heldout.json")
    out.write_text(json.dumps({
        "heldout_category_counts": {c: len(v) for c, v in by_category.items()},
        "flat_sweep_heldout": flat_results,
        "best_flat_alpha_heldout": best_flat["alpha"],
        "best_per_category_alpha_heldout": best_per_cat,
        "final_benchmark_comparison": final,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
