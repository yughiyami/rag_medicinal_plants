"""
CRAG stress test with ABSOLUTE-threshold evaluator (reviewer O1).

The default CRAGEvaluator normalizes cross-encoder scores within each batch
(min-max), which makes the fixed accept/refine thresholds relative to the
worst document in the same query — so any query with a mediocre-but-best
document still triggers `accept`. This is why the original benchmark
reports 100% `accept` and the corrective branches never fire.

This runner replaces the within-batch normalization with an ABSOLUTE
sigmoid transform of the raw cross-encoder logit — a standard calibration
choice for MS MARCO cross-encoders. The absolute threshold then reflects
true query-document relevance and the corrective branches fire on
out-of-distribution probes as designed.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval.hybrid import HybridRetriever
from agent.crag_evaluator import CRAGDecision

from run_crag_stress_test import build_probe_set  # reuse probe set


ACCEPT_TH = 0.60   # sigmoid-space accept
PARTIAL_TH = 0.30  # sigmoid-space refine
MIN_RELEVANT_RATIO = 0.2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_absolute(query, results):
    if not results:
        return CRAGDecision(action="web_search", confidence=1.0, reason="No documents")

    raw = np.array([r.get("rerank_score", 0.0) or 0.0 for r in results], dtype=float)
    scores = sigmoid(raw)

    relevant = np.where(scores >= ACCEPT_TH)[0].tolist()
    partial = np.where((scores >= PARTIAL_TH) & (scores < ACCEPT_TH))[0].tolist()

    relevant_ratio = len(relevant) / len(results)
    max_s = float(scores.max())

    if relevant_ratio >= MIN_RELEVANT_RATIO and len(relevant) >= 1:
        return CRAGDecision(
            action="accept",
            confidence=float(scores[relevant].mean()),
            relevant_indices=relevant,
            reason=f"{len(relevant)}/{len(results)} above accept ({ACCEPT_TH})",
        )
    if partial:
        return CRAGDecision(
            action="refine",
            confidence=float(scores[partial].mean()),
            relevant_indices=partial,
            reason=f"{len(partial)} partially relevant, max={max_s:.3f}",
        )
    return CRAGDecision(
        action="web_search",
        confidence=float(1.0 - max_s),
        reason=f"max relevance {max_s:.3f} below partial threshold {PARTIAL_TH}",
    )


def main():
    print("=" * 70)
    print(f"CRAG stress test — ABSOLUTE thresholds (accept>={ACCEPT_TH}, "
          f"refine>={PARTIAL_TH})")
    print("=" * 70)

    retriever = HybridRetriever()
    retriever.load()

    probes = build_probe_set()
    print(f"Probe set: {len(probes)} queries")

    results = []
    t0 = time.time()
    for i, probe in enumerate(probes, 1):
        q = probe["query"]
        try:
            docs = retriever.retrieve(q, top_k=30, rerank_top_k=10)
            decision = evaluate_absolute(q, docs)
            raw = [float(d.get("rerank_score", 0.0) or 0.0) for d in docs] if docs else []
            s_max_sig = float(sigmoid(np.array(raw)).max()) if raw else 0.0
        except Exception as e:
            docs, decision, s_max_sig = [], None, 0.0
            probe["error"] = str(e)

        results.append({
            **probe,
            "action": decision.action if decision else "error",
            "confidence": float(decision.confidence) if decision else 0.0,
            "s_max_sigmoid": s_max_sig,
            "n_docs": len(docs),
        })
        print(f"  [{i:02d}/{len(probes)}] {probe['family']:20s} "
              f"action={results[-1]['action']:11s} sig(max)={s_max_sig:.3f}  "
              f"q={q[:60]}")

    print(f"\nElapsed: {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print("Summary by family — ABSOLUTE thresholds")
    print("=" * 70)
    families = {}
    for r in results:
        families.setdefault(r["family"], []).append(r["action"])
    for fam, actions in families.items():
        acc = actions.count("accept")
        ref = actions.count("refine")
        web = actions.count("web_search")
        err = actions.count("error")
        print(f"  {fam:22s}  accept={acc:2d}  refine={ref:2d}  "
              f"web_search={web:2d}  error={err:2d}  n={len(actions)}")

    out = Path("results") / "crag_stress_absolute.json"
    out.write_text(json.dumps({
        "config": {
            "accept_threshold_sigmoid": ACCEPT_TH,
            "partial_threshold_sigmoid": PARTIAL_TH,
            "min_relevant_ratio": MIN_RELEVANT_RATIO,
            "normalization": "sigmoid(raw_ce_score)",
        },
        "probes": results,
        "summary": {fam: {
            "accept": actions.count("accept"),
            "refine": actions.count("refine"),
            "web_search": actions.count("web_search"),
            "error": actions.count("error"),
            "n": len(actions),
        } for fam, actions in families.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
