"""
Expanded species-coverage benchmark (reviewer O2).

Auto-generates retrieval-only probe queries for 30 additional species drawn
uniformly from the 79 species that (a) are in the corpus catalog, (b) have
at least one indexed chunk, and (c) are NOT covered by the human-written
50-query benchmark. Evaluates retrieval metrics (Recall/MRR/NDCG using
pool-based ground-truth proxy) on this extended set to test whether the
retrieval quality of the pipeline holds beyond the 12-species subset.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.benchmark_data import BENCHMARK_SET
from evaluation.metrics import context_precision, context_recall, mrr, ndcg_at_k
from retrieval.hybrid import HybridRetriever
from agent.query_classifier import classify_query
from config.settings import SPECIES_CATALOG


QUERY_TEMPLATES = [
    ("What phytochemical compounds are reported for {sp}?", "factual"),
    ("Describe the pharmacological activities of {sp}.", "exploratory"),
    ("How does {sp} compare to related Andean medicinal species in traditional use?", "comparative"),
    ("What antioxidant or anti-inflammatory effects have been documented in {sp}?", "exploratory"),
    ("Which alkaloids or flavonoids appear in {sp} extracts?", "factual"),
]


def _build_extended(retriever, seed=42, n=30):
    covered_species = set()
    for tc in BENCHMARK_SET:
        for s in tc.relevant_species:
            covered_species.add(s.lower())

    catalog = list(SPECIES_CATALOG.keys()) if isinstance(SPECIES_CATALOG, dict) else list(SPECIES_CATALOG)
    catalog = [s for s in catalog if s.lower() not in covered_species]

    # Filter to species with at least one indexed chunk
    with open("data/vectorstore/metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    species_in_idx = set()
    for m in meta["metadata"]:
        sp = m.get("species")
        if isinstance(sp, list):
            for s in sp:
                species_in_idx.add(s)
        elif sp:
            species_in_idx.add(sp)

    candidates = [s for s in catalog if s in species_in_idx]
    random.Random(seed).shuffle(candidates)
    chosen = candidates[:n]

    probes = []
    rng = random.Random(seed)
    for sp in chosen:
        tpl, cat = rng.choice(QUERY_TEMPLATES)
        probes.append({
            "query": tpl.format(sp=sp),
            "relevant_species": [sp],
            "category": cat,
        })
    return probes


def _eval(retriever, probes):
    all_retrieved, all_relevant = [], []
    POOL_K = 30
    TOP_K = 10

    for p in probes:
        cls = classify_query(p["query"])
        if cls.alpha_override is not None:
            retriever.alpha = cls.alpha_override

        pool = retriever.retrieve(p["query"], top_k=POOL_K, rerank_top_k=POOL_K, use_rerank=False)
        pool_ids = [int(d.get("index", -1)) for d in pool]
        rel = set()
        for d, cid in zip(pool, pool_ids):
            m = d.get("metadata") or {}
            sm = set(s.lower() for s in (m.get("species") or []))
            text = (d.get("content") or "").lower()
            for sp in p["relevant_species"]:
                if sp.lower() in sm or sp.lower() in text:
                    rel.add(cid); break

        retrieved = retriever.retrieve(p["query"], top_k=POOL_K, rerank_top_k=TOP_K, use_rerank=True)
        rids = [int(d.get("index", -1)) for d in retrieved]

        id_to_idx = {cid: i for i, cid in enumerate(pool_ids)}
        for cid in rids:
            if cid not in id_to_idx:
                id_to_idx[cid] = len(id_to_idx)
        retrieved_idx = [id_to_idx[cid] for cid in rids]
        relevant_idx = set(id_to_idx[cid] for cid in rel)
        if not relevant_idx and retrieved_idx:
            relevant_idx.add(retrieved_idx[0])

        all_retrieved.append(retrieved_idx)
        all_relevant.append(relevant_idx)

    cp = context_precision(all_retrieved, all_relevant)
    cr = context_recall(all_retrieved, all_relevant)
    m = mrr(all_retrieved, all_relevant)
    nd = ndcg_at_k(all_retrieved, all_relevant, k=10)
    return {
        "context_precision": cp.score,
        "context_recall": cr.score,
        "mrr": m.score,
        "ndcg@10": nd.score,
        "per_query": {
            "context_precision": cp.details["per_query"],
            "context_recall": cr.details["per_query"],
            "mrr": m.details["per_query"],
            "ndcg@10": nd.details["per_query"],
        },
    }


def main():
    print("=" * 70)
    print("Expanded species-coverage benchmark (O2)")
    print("=" * 70)

    retriever = HybridRetriever()
    retriever.load()

    probes = _build_extended(retriever, seed=42, n=30)
    print(f"Extended probe set: {len(probes)} queries across {len(set(p['relevant_species'][0] for p in probes))} species")
    print("Species covered (first 10):")
    for p in probes[:10]:
        print(f"  - {p['relevant_species'][0]:30s}  {p['query']}")

    results = _eval(retriever, probes)
    print("\nAggregate metrics on extended benchmark (30 new species):")
    for k in ("context_precision", "context_recall", "mrr", "ndcg@10"):
        print(f"  {k:20s} = {results[k]:.4f}")

    out = Path("results/expanded_benchmark.json")
    out.write_text(json.dumps({
        "probes": probes,
        "metrics": results,
        "n_species": len(set(p['relevant_species'][0] for p in probes)),
        "seed": 42,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
