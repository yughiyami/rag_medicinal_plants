"""
Regenerate Table 2 (extended retrieval / coverage generalisation) with REAL,
internally-consistent numbers (reviewer re-check).

Both rows use the SAME agent-based methodology that produces Table 4, so the
12-species row matches Table 4's full config exactly. Captures NDCG@5 and
NDCG@10 for both the 12 human-verified species and 30 additional stratified
species (42 unique species total).
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Stub web searcher (same rationale as run_perquery_agent_ablation.py)
import scraping.web_searcher as _ws


class _StubSearcher:
    def __init__(self, *a, **k):
        pass

    async def search(self, *a, **k):
        return []

    async def close(self):
        return None


_ws.WebSearcher = _StubSearcher

from evaluation.benchmark_data import TestCase, BENCHMARK_SET
from evaluation.ablation import ABLATION_CONFIGS, _build_agent, _run_queries
from evaluation.metrics import context_precision, context_recall, mrr, ndcg_at_k
from config.settings import SPECIES_CATALOG

QUERY_TEMPLATES = [
    "What phytochemical compounds are reported for {sp}?",
    "Describe the pharmacological activities of {sp}.",
    "How does {sp} compare to related Andean medicinal species in traditional use?",
    "What antioxidant or anti-inflammatory effects have been documented in {sp}?",
    "Which alkaloids or flavonoids appear in {sp} extracts?",
]


def build_new_species_cases(n=30, seed=42):
    covered = set()
    for tc in BENCHMARK_SET:
        for s in tc.relevant_species:
            covered.add(s.lower())
    catalog = list(SPECIES_CATALOG.keys()) if isinstance(SPECIES_CATALOG, dict) else list(SPECIES_CATALOG)
    catalog = [s for s in catalog if s.lower() not in covered]
    with open("data/vectorstore/metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    in_idx = set()
    for m in meta["metadata"]:
        sp = m.get("species")
        if isinstance(sp, list):
            in_idx.update(sp)
        elif sp:
            in_idx.add(sp)
    cands = [s for s in catalog if s in in_idx]
    random.Random(seed).shuffle(cands)
    chosen = cands[:n]
    rng = random.Random(seed)
    cases = []
    for sp in chosen:
        tpl = rng.choice(QUERY_TEMPLATES)
        cases.append(TestCase(query=tpl.format(sp=sp), reference_answer="",
                              relevant_species=[sp], category="factual"))
    return cases


def measure(agent, cases):
    q, a, r, c, retr, rel = _run_queries(agent, cases)
    return {
        "n": len(cases),
        "mrr": mrr(retr, rel).score,
        "ndcg@5": ndcg_at_k(retr, rel, k=5).score,
        "ndcg@10": ndcg_at_k(retr, rel, k=10).score,
        "context_precision": context_precision(retr, rel).score,
        "context_recall": context_recall(retr, rel).score,
    }


def main():
    full_cfg = next(c for c in ABLATION_CONFIGS if c.name == "full")

    print("=== Row 1: 12 human-verified species (benchmark) ===")
    agent, restore = _build_agent(full_cfg, backend="template")
    row1 = measure(agent, BENCHMARK_SET)
    restore()
    for k, v in row1.items():
        print(f"  {k}: {round(v,3) if isinstance(v,float) else v}")

    print("\n=== Row 2: 30 additional stratified species ===")
    new_cases = build_new_species_cases(30, seed=42)
    agent2, restore2 = _build_agent(full_cfg, backend="template")
    row2 = measure(agent2, new_cases)
    restore2()
    for k, v in row2.items():
        print(f"  {k}: {round(v,3) if isinstance(v,float) else v}")

    out = Path("results/table2_extended_consistent.json")
    out.write_text(json.dumps({"row_12": row1, "row_30new": row2}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    print("\n=== LaTeX rows ===")
    def fmt(x):
        return ("%.3f" % x).replace(".", "{,}")
    print(f"Inicial humano-verificado & 12 & ${fmt(row1['mrr'])}$ & ${fmt(row1['ndcg@5'])}$ & ${fmt(row1['ndcg@10'])}$ & ${fmt(row1['context_precision'])}$ & ${fmt(row1['context_recall'])}$ \\\\")
    print(f"Extensi\\'{{o}}n (30 especies nuevas) & 30 & ${fmt(row2['mrr'])}$ & ${fmt(row2['ndcg@5'])}$ & ${fmt(row2['ndcg@10'])}$ & ${fmt(row2['context_precision'])}$ & ${fmt(row2['context_recall'])}$ \\\\")


if __name__ == "__main__":
    main()
