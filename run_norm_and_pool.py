"""
Two reviewer items in a single retrieval pass (no API calls).

R2-1c  within-batch min-max normalization vs absolute sigmoid, on the SAME
       27 stress probes and the SAME 50 in-domain queries. The min-max
       implementation is the original one, recovered verbatim from commit
       6c6f93c of agent/crag_evaluator.py (it was replaced by the sigmoid in
       f5f6f54, so the current code can no longer reproduce the old arm).
R2-2a  incremental value of hybrid fusion measured on the top-30 candidate
       POOL, before the cross-encoder collapses the configurations.

Ground truth for the pool metrics is species matching over the union of
dense@30 and sparse@30, so it does not depend on which configuration is being
scored (the repo's own harness builds it from the hybrid pool only).

Usage: python run_norm_and_pool.py --store <vectorstore_dir> --out <json>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = r"D:\BaseAvanzado\sirca_rag"
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.abspath("./hf_cache"))

from retrieval.hybrid import HybridRetriever            # noqa: E402
from evaluation.benchmark_data import BENCHMARK_SET     # noqa: E402
from run_crag_stress_test import build_probe_set        # noqa: E402
from evaluation.metrics import context_precision, context_recall, mrr, ndcg_at_k  # noqa: E402

ACCEPT, PARTIAL, MIN_RATIO = 0.60, 0.30, 0.2
POOL_K = 30
RERANK_K = 10


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.asarray(s, dtype=float)))


def minmax(s):
    """Original within-batch normalization (commit 6c6f93c)."""
    s = np.asarray(s, dtype=float)
    if len(s) == 0:
        return s
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-6:
        return np.full_like(s, 0.5)
    return (s - lo) / (hi - lo)


def decide(norm):
    """Downstream CRAG decision logic, identical for both normalizations."""
    if len(norm) == 0:
        return "web_search"
    rel = np.where(norm >= ACCEPT)[0]
    par = np.where((norm >= PARTIAL) & (norm < ACCEPT))[0]
    if len(rel) / len(norm) >= MIN_RATIO and len(rel) >= 1:
        return "accept"
    if len(par) > 0:
        return "refine"
    return "web_search"


def species_relevant(docs, ids, species):
    rel = set()
    for d, cid in zip(docs, ids):
        meta = d.get("metadata") or {}
        meta_sp = {s.lower() for s in (meta.get("species") or [])}
        text = (d.get("content") or "").lower()
        for sp in species:
            spl = sp.lower()
            if spl in meta_sp or spl in text:
                rel.add(cid)
                break
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = HybridRetriever(vectorstore_path=Path(args.store))
    r.load()

    rows = []
    pool_rows = []
    t0 = time.time()

    queries = [{"kind": "probe", "family": p["family"], "query": p["query"],
                "expected_route": p["expected_route"], "species": []}
               for p in build_probe_set()]
    queries += [{"kind": "benchmark", "family": tc.category, "query": tc.query,
                 "expected_route": None, "species": list(tc.relevant_species)}
                for tc in BENCHMARK_SET]

    for i, q in enumerate(queries, 1):
        query = q["query"]
        dense = r._dense_search(query, POOL_K)
        sparse = r._sparse_search(query, POOL_K)

        r.alpha = 0.6
        fused = r._reciprocal_rank_fusion(dense, sparse)
        cands = [dict(c) for c in fused[:POOL_K]]
        top10 = r._rerank(query, cands, RERANK_K)
        raw = [float(d["rerank_score"]) for d in top10]

        n_sig = sigmoid(raw)
        n_mm = minmax(raw)
        rows.append({
            "kind": q["kind"], "family": q["family"], "query": query,
            "expected_route": q["expected_route"],
            "raw_rerank_scores": raw,
            "action_sigmoid": decide(n_sig),
            "action_minmax": decide(n_mm),
            "s_max_sigmoid": float(n_sig.max()) if len(raw) else 0.0,
            "n_above_accept_sigmoid": int((n_sig >= ACCEPT).sum()),
            "n_above_accept_minmax": int((n_mm >= ACCEPT).sum()),
            "top10_ids": [int(d.get("index", -1)) for d in top10],
        })

        if q["kind"] == "benchmark" and q["species"]:
            union = dense + [s for s in sparse
                             if s["index"] not in {d["index"] for d in dense}]
            union_ids = [int(d["index"]) for d in union]
            rel = species_relevant(union, union_ids, q["species"])

            configs = {}
            configs["dense_only"] = [int(d["index"]) for d in dense[:POOL_K]]
            configs["sparse_only"] = [int(d["index"]) for d in sparse[:POOL_K]]
            configs["hybrid_a06"] = [int(d["index"]) for d in fused[:POOL_K]]
            pool_rows.append({"query": query, "n_relevant_union": len(rel),
                              "relevant": sorted(rel), "configs": configs})

        if i % 10 == 0:
            print(f"  [{i}/{len(queries)}] {time.time()-t0:.0f}s", flush=True)

    # ---- pool-level aggregate metrics (same metric code as the paper) ----
    pool_metrics = {}
    for cfg in ("dense_only", "sparse_only", "hybrid_a06"):
        retr = [p["configs"][cfg] for p in pool_rows]
        relv = [set(p["relevant"]) for p in pool_rows]
        pool_metrics[cfg] = {
            "context_precision": context_precision(retr, relv).score,
            "context_recall@30": context_recall(retr, relv).score,
            "mrr": mrr(retr, relv).score,
            "ndcg@10": ndcg_at_k(retr, relv, k=10).score,
            "n_queries": len(retr),
        }

    # ---- normalization contingency ----
    def tally(kind):
        sub = [x for x in rows if x["kind"] == kind]
        out = {}
        for arm in ("action_sigmoid", "action_minmax"):
            out[arm] = {a: sum(1 for x in sub if x[arm] == a)
                        for a in ("accept", "refine", "web_search")}
        out["n"] = len(sub)
        return out

    summary = {
        "store": args.store,
        "thresholds": {"accept": ACCEPT, "partial": PARTIAL,
                       "min_relevant_ratio": MIN_RATIO},
        "probes": tally("probe"),
        "benchmark": tally("benchmark"),
        "probes_by_family": {
            fam: {arm: {a: sum(1 for x in rows
                               if x["kind"] == "probe" and x["family"] == fam
                               and x[arm] == a)
                        for a in ("accept", "refine", "web_search")}
                  for arm in ("action_sigmoid", "action_minmax")}
            for fam in sorted({x["family"] for x in rows if x["kind"] == "probe"})
        },
        "pool_metrics": pool_metrics,
        "elapsed_s": round(time.time() - t0, 1),
    }

    Path(args.out).write_text(json.dumps(
        {"summary": summary, "per_query": rows, "pool_per_query": pool_rows},
        indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
