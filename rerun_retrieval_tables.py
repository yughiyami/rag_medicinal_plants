"""
Re-run the paper's retrieval-side tables against an arbitrary vectorstore.

Reproduces run_perquery_agent_ablation.py (Tables 3 retrieval + 4) and
run_table2_extended.py (Table 1) without touching the repo: the vectorstore
directory is patched in retrieval.hybrid, the web searcher is stubbed (no
network), and the generator runs on the template backend (no API calls), which
is what the repo's own runners do — retrieval metrics are backend-independent.

Usage: python rerun_retrieval_tables.py --store <dir> --tag <name>
Writes  ablation_<tag>.json  and  table1_<tag>.json  in the current directory.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(r"D:\BaseAvanzado\sirca_rag")
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()

# 1) point the retriever at the requested store BEFORE anything instantiates it
import retrieval.hybrid as H                                   # noqa: E402
H.VECTORSTORE_DIR = Path(args.store).resolve()

# 2) stub the web searcher (same rationale as the repo's runners)
import scraping.web_searcher as _ws                            # noqa: E402


class _StubSearcher:
    def __init__(self, *a, **k):
        pass

    async def search(self, *a, **k):
        return []

    async def close(self):
        return None


_ws.WebSearcher = _StubSearcher

from evaluation.ablation import ABLATION_CONFIGS, _build_agent, _run_queries   # noqa: E402
from evaluation.benchmark_data import TestCase, BENCHMARK_SET                  # noqa: E402
from evaluation.metrics import context_precision, context_recall, mrr, ndcg_at_k  # noqa: E402
from config.settings import SPECIES_CATALOG                                    # noqa: E402
from scipy.stats import wilcoxon                                               # noqa: E402

QUERY_TEMPLATES = [
    "What phytochemical compounds are reported for {sp}?",
    "Describe the pharmacological activities of {sp}.",
    "How does {sp} compare to related Andean medicinal species in traditional use?",
    "What antioxidant or anti-inflammatory effects have been documented in {sp}?",
    "Which alkaloids or flavonoids appear in {sp} extracts?",
]


def build_new_species_cases(n=30, seed=42):
    """Same selection as run_table2_extended.py, reading the patched store."""
    covered = {s.lower() for tc in BENCHMARK_SET for s in tc.relevant_species}
    catalog = (list(SPECIES_CATALOG.keys()) if isinstance(SPECIES_CATALOG, dict)
               else list(SPECIES_CATALOG))
    catalog = [s for s in catalog if s.lower() not in covered]
    meta = json.loads((H.VECTORSTORE_DIR / "metadata.json").read_text(encoding="utf-8"))
    in_idx = set()
    for m in meta["metadata"]:
        sp = m.get("species")
        if isinstance(sp, list):
            in_idx.update(sp)
        elif sp:
            in_idx.add(sp)
    cands = [s for s in catalog if s in in_idx]
    random.Random(seed).shuffle(cands)
    rng = random.Random(seed)
    return [TestCase(query=rng.choice(QUERY_TEMPLATES).format(sp=sp),
                     reference_answer="", relevant_species=[sp], category="factual")
            for sp in cands[:n]]


def score(retrieved, relevant):
    cp = context_precision(retrieved, relevant)
    cr = context_recall(retrieved, relevant)
    mr = mrr(retrieved, relevant)
    n5 = ndcg_at_k(retrieved, relevant, k=5)
    n10 = ndcg_at_k(retrieved, relevant, k=10)
    return cp, cr, mr, n5, n10


# ---------- Tables 3 (retrieval) and 4: five-configuration ablation ----------
per_query, aggregate = {}, {}
for cfg in ABLATION_CONFIGS:
    t0 = time.time()
    agent, restore = _build_agent(cfg, backend="template")
    _, _, _, _, retrieved, relevant = _run_queries(agent, BENCHMARK_SET)
    cp, cr, mr, _, n10 = score(retrieved, relevant)
    per_query[cfg.name] = {"context_precision": cp.details["per_query"],
                           "context_recall": cr.details["per_query"],
                           "mrr": mr.details["per_query"],
                           "ndcg@10": n10.details["per_query"]}
    aggregate[cfg.name] = {"context_precision": cp.score, "context_recall": cr.score,
                           "mrr": mr.score, "ndcg@10": n10.score}
    print(f"[{cfg.name}] CP={cp.score:.3f} CR={cr.score:.3f} MRR={mr.score:.3f} "
          f"NDCG@10={n10.score:.3f} ({time.time()-t0:.0f}s)", flush=True)
    restore()

wil = {}
for name in per_query:
    if name == "full":
        continue
    wil[name] = {}
    for metric in ("context_precision", "context_recall", "mrr", "ndcg@10"):
        a = per_query["full"][metric]
        b = per_query[name][metric]
        diff = [x - y for x, y in zip(a, b)]
        nz = sum(1 for d in diff if d != 0)
        if nz == 0:
            wil[name][metric] = {"p": None, "n_nonzero": 0}
        else:
            st, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            wil[name][metric] = {"p": float(p), "statistic": float(st), "n_nonzero": nz}

Path(f"ablation_{args.tag}.json").write_text(json.dumps(
    {"store": str(H.VECTORSTORE_DIR), "aggregate": aggregate,
     "per_query": per_query, "wilcoxon_vs_full": wil}, indent=2), encoding="utf-8")

# ---------- Table 1: 12 human-verified species vs 30 additional species ----------
agent, restore = _build_agent(ABLATION_CONFIGS[0], backend="template")
rows = {}
for label, cases in (("row_12_human_verified", BENCHMARK_SET),
                     ("row_30new_species", build_new_species_cases(30, seed=42))):
    _, _, _, _, retrieved, relevant = _run_queries(agent, cases)
    cp, cr, mr, n5, n10 = score(retrieved, relevant)
    rows[label] = {"n_queries": len(cases), "mrr": mr.score, "ndcg@5": n5.score,
                   "ndcg@10": n10.score, "context_precision": cp.score,
                   "context_recall": cr.score}
    print(f"[table1:{label}] n={len(cases)} MRR={mr.score:.3f} NDCG@10={n10.score:.3f} "
          f"CP={cp.score:.3f} CR={cr.score:.3f}", flush=True)
restore()

Path(f"table1_{args.tag}.json").write_text(json.dumps(
    {"store": str(H.VECTORSTORE_DIR), **rows}, indent=2), encoding="utf-8")
print("done", flush=True)
