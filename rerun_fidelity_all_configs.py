"""
Close Limitation (iii): per-query Fidelity for the three ablation configurations
that Table 3 reports as "n/a" (dense_only, sparse_only, no_crag), over the same
n=80 query set and with the same DeepSeek generator used for full and
no_reranker, so the whole column becomes comparable.

Each configuration is written to disk as soon as it finishes.

Usage: python rerun_fidelity_all_configs.py --store <dir> --tag <name>
"""
import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

REPO = Path(r"D:\BaseAvanzado\sirca_rag")
sys.path.insert(0, str(REPO))
os.chdir(REPO)

_stub = types.ModuleType("dotenv")
_stub.load_dotenv = lambda *a, **k: False
_stub.find_dotenv = lambda *a, **k: ""
sys.modules["dotenv"] = _stub

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()
OUT, STORE = Path(args.out_dir), Path(args.store).resolve()
assert os.environ.get("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY missing"

import retrieval.hybrid as H                                    # noqa: E402
H.VECTORSTORE_DIR = STORE

import scraping.web_searcher as _ws                             # noqa: E402


class _StubSearcher:
    def __init__(self, *a, **k):
        pass

    async def search(self, *a, **k):
        return []

    async def close(self):
        return None


_ws.WebSearcher = _StubSearcher

from evaluation.benchmark_data import TestCase, BENCHMARK_SET    # noqa: E402
from evaluation.ablation import ABLATION_CONFIGS, _build_agent, _run_queries  # noqa: E402
from evaluation.metrics import faithfulness                      # noqa: E402
from config.settings import SPECIES_CATALOG                      # noqa: E402
from scipy.stats import wilcoxon                                 # noqa: E402

QUERY_TEMPLATES = [
    "What phytochemical compounds are reported for {sp}?",
    "Describe the pharmacological activities of {sp}.",
    "How does {sp} compare to related Andean medicinal species in traditional use?",
    "What antioxidant or anti-inflammatory effects have been documented in {sp}?",
    "Which alkaloids or flavonoids appear in {sp} extracts?",
]


def build_new_species_cases(n=30, seed=42):
    covered = {s.lower() for tc in BENCHMARK_SET for s in tc.relevant_species}
    catalog = (list(SPECIES_CATALOG.keys()) if isinstance(SPECIES_CATALOG, dict)
               else list(SPECIES_CATALOG))
    catalog = [s for s in catalog if s.lower() not in covered]
    meta = json.loads((STORE / "metadata.json").read_text(encoding="utf-8"))
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


def patch_deepseek_budget():
    """Escalate max_tokens on empty content (see Limitation vii)."""
    import generation.grounded_generator as G

    def _escalating(self, prompt, max_tokens):
        from openai import OpenAI
        client = OpenAI(api_key=G.DEEPSEEK_API_KEY, base_url=G.DEEPSEEK_BASE_URL)
        last = None
        for budget in (max_tokens, max_tokens * 2, max_tokens * 4):
            for attempt in range(3):
                try:
                    r = client.chat.completions.create(
                        model=G.DEEPSEEK_MODEL,
                        messages=[{"role": "system", "content": G.SYSTEM_PROMPT},
                                  {"role": "user", "content": prompt}],
                        max_tokens=budget, temperature=G.TEMPERATURE, top_p=0.9)
                except Exception as e:
                    last = e
                    time.sleep(2.0 * (attempt + 1))
                    continue
                ans = (r.choices[0].message.content or "").strip()
                if ans:
                    return G.GenerationResult(
                        answer=ans,
                        citations_used=G._extract_citation_indices(ans),
                        model=G.DEEPSEEK_MODEL,
                        tokens_generated=r.usage.completion_tokens)
                last = RuntimeError(f"empty at {budget}")
                break
        raise RuntimeError(f"DeepSeek empty after escalation: {last}")

    G.GroundedGenerator._deepseek_generate = _escalating


patch_deepseek_budget()
CASES = BENCHMARK_SET + build_new_species_cases(30, seed=42)
print(f"store={STORE}  n={len(CASES)}", flush=True)

out_path = OUT / f"fidelity_all_configs_{args.tag}.json"
res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

for name in ("dense_only", "sparse_only", "no_crag"):
    if name in res:
        print(f"[{name}] already present, skipping", flush=True)
        continue
    cfg = next(c for c in ABLATION_CONFIGS if c.name == name)
    agent, restore = _build_agent(cfg, backend="deepseek")
    t0 = time.time()
    _, answers, _, contexts, _, _ = _run_queries(agent, CASES)
    restore()
    ff = faithfulness(answers, contexts)
    res[name] = {"per_query": ff.details["per_sample"], "mean": ff.score,
                 "n_blank": sum(1 for a in answers if not a.strip()),
                 "elapsed_s": round(time.time() - t0, 1)}
    print(f"[{name}] Fidelity={ff.score:.4f} blank={res[name]['n_blank']} "
          f"({res[name]['elapsed_s']:.0f}s)", flush=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

# Wilcoxon against full, reusing the stored full run
full = json.loads((OUT / f"n5_fidelity_wilcoxon_{args.tag}.json").read_text(
    encoding="utf-8"))["full_fidelity_per_query"]
for name in ("dense_only", "sparse_only", "no_crag"):
    a, b = full, res[name]["per_query"]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    nz = sum(1 for x, y in zip(a, b) if x != y)
    _, p2 = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    fm = sum(a) / n
    res[name]["vs_full"] = {"p_two_sided": float(p2), "n_nonzero": nz, "n": n,
                            "delta": res[name]["mean"] - fm,
                            "delta_rel_pct": (res[name]["mean"] - fm) / fm * 100}
    print(f"[{name}] vs full: delta={res[name]['vs_full']['delta']:+.4f} "
          f"({res[name]['vs_full']['delta_rel_pct']:+.1f}%) p={p2:.4f} "
          f"n_nonzero={nz}/{n}", flush=True)

out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("done", flush=True)
