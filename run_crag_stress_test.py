"""
CRAG corrective-branch stress test (reviewer O1).

Runs three families of deliberately out-of-distribution queries through the
retrieval + CRAG stage and reports the routing decision (accept / refine /
web_search) and the maximum reranker score s_max, so the corrective branches
can be shown to activate on the intended inputs.

Families:
  A. Missing-species probes — 9 queries about species catalogued but absent
     from the vector index (no chunks). Expected: refine or web_search.
  B. Off-domain probes — 10 queries about topics entirely outside ethnobotany.
     Expected: web_search.
  C. Garbled / adversarial probes — 8 queries with malformed language.
     Expected: refine.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval.hybrid import HybridRetriever
from agent.crag_evaluator import CRAGEvaluator


MISSING_SPECIES = [
    "Adesmia spinosissima",
    "Alternanthera porrigens",
    "Berberis lutea",
    "Diplostephium tovarii",
    "Gentiana sedifolia",
    "Junellia minima",
    "Lepidium chichicara",
    "Margiricarpus pinnatus",
    "Senecio canescens",
]


def build_probe_set():
    probes = []

    for sp in MISSING_SPECIES:
        probes.append({
            "family": "A_missing_species",
            "query": f"What phytochemical compounds are reported for {sp}?",
            "expected_route": "refine_or_web_search",
        })

    off_domain = [
        "How do I configure a Kubernetes deployment for a Node.js service?",
        "What is the current inflation rate in Argentina for 2026?",
        "Explain the difference between quicksort and mergesort algorithms.",
        "Summarise the plot of the novel Cien Anos de Soledad.",
        "What are the visa requirements for travelling from Peru to Japan?",
        "How does the Fourier transform apply to audio signal processing?",
        "Describe the training regime of an Olympic marathon runner.",
        "What is the chemical composition of stainless steel 316?",
        "Compare Vue and Svelte for building a modern SPA.",
        "How is Bitcoin's proof-of-work consensus algorithm structured?",
    ]
    for q in off_domain:
        probes.append({"family": "B_off_domain", "query": q, "expected_route": "web_search"})

    garbled = [
        "medicinal plant peru ??? xxx antioxidant ????",
        "uncaria uncaria uncaria alkaloid alkaloid",
        "plant plant plant plant plant",
        "asdklj laskdj alskdj plantas",
        "what is the of the by which under a certain",
        "peruvian andean andean andean andean",
        "give me the something interesting maybe",
        "medicinal use compound activity",
    ]
    for q in garbled:
        probes.append({"family": "C_garbled", "query": q, "expected_route": "refine"})

    return probes


def main():
    print("=" * 70)
    print("CRAG corrective-branch stress test")
    print("=" * 70)

    retriever = HybridRetriever()
    retriever.load()
    evaluator = CRAGEvaluator()

    probes = build_probe_set()
    print(f"Probe set: {len(probes)} queries "
          f"(A={sum(p['family']=='A_missing_species' for p in probes)}, "
          f"B={sum(p['family']=='B_off_domain' for p in probes)}, "
          f"C={sum(p['family']=='C_garbled' for p in probes)})")

    results = []
    t0 = time.time()
    for i, probe in enumerate(probes, 1):
        q = probe["query"]
        try:
            docs = retriever.retrieve(q, top_k=30, rerank_top_k=10)
            decision = evaluator.evaluate(q, docs)
            s_max = float(max((d.get("rerank_score", 0.0) or 0.0) for d in docs)) if docs else 0.0
        except Exception as e:
            docs, decision, s_max = [], None, 0.0
            probe["error"] = str(e)

        results.append({
            **probe,
            "action": decision.action if decision else "error",
            "confidence": float(decision.confidence) if decision else 0.0,
            "s_max": s_max,
            "n_docs": len(docs),
        })
        print(f"  [{i:02d}/{len(probes)}] {probe['family']:20s} "
              f"action={results[-1]['action']:12s} s_max={s_max:.3f}  "
              f"q={q[:60]}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("Summary: routing distribution per family")
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

    out = Path("results") / "crag_stress_test.json"
    out.write_text(json.dumps({
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
