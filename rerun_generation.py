"""
Re-run the paper's generation-side experiments against an arbitrary vectorstore.

Stages (selectable with --only), each written to disk as soon as it finishes so
a later failure does not lose earlier work:

  xllm      cross-LLM benchmark (Table 2) — DeepSeek V4-Flash vs the Cerebras
            model given by --cerebras-model, on the 50-query benchmark, all
            five generation metrics + paired t-tests.
  judge     LLM-as-judge (Sec. 5.2) — the Cerebras model grades DeepSeek's
            answers claim-level; correlate with algorithmic Fidelity.
  fidelity  Fidelity significance (Table 3 / Sec. 5.4) — full vs no_reranker
            over n=80 (50 benchmark + 30 template species), paired Wilcoxon.
            DeepSeek only, so it runs without Cerebras.

The repo's own prompts and DeepSeek caller are imported and reused verbatim so
the numbers stay comparable with the published run. Differences from the repo
runners: the vectorstore directory is patched, API keys come from the
environment (never from .env), the Cerebras model id is a parameter, and
outputs are tagged and written outside the repo instead of overwriting
results/.

Usage:
  python rerun_generation.py --store <dir> --tag <name> [--only fidelity]
                             [--cerebras-model qwen-3.8-27b]
"""
import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(r"D:\BaseAvanzado\sirca_rag")
sys.path.insert(0, str(REPO))
os.chdir(REPO)                      # repo runners use relative model/data paths

# config/settings.py calls load_dotenv(.env) guarded only against ImportError.
# .env is unreadable here (the sandbox denies protected config paths) and
# raises PermissionError instead, which that guard does not catch. The keys
# already come from the environment, so neutralise the loader with a no-op.
_stub = types.ModuleType("dotenv")
_stub.load_dotenv = lambda *a, **k: False
_stub.find_dotenv = lambda *a, **k: ""
sys.modules["dotenv"] = _stub

ap = argparse.ArgumentParser()
ap.add_argument("--store", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--only", default="xllm,judge,fidelity")
ap.add_argument("--gen2-provider", default="sambanova",
                choices=("sambanova", "cerebras"),
                help="who serves the second generator. Cerebras no longer "
                     "serves gemma-4-31b; SambaNova serves it as gemma-4-31B-it.")
ap.add_argument("--gen2-model", default=None,
                help="model id at that provider (default: the provider's "
                     "spelling of Gemma-4-31B)")
args = ap.parse_args()
STAGES = {s.strip() for s in args.only.split(",") if s.strip()}

GEN2 = {
    "sambanova": {"url": "https://api.sambanova.ai/v1/chat/completions",
                  "key_env": "SAMBA_APIKEY", "model": "gemma-4-31B-it",
                  "label": "Gemma-4-31B (SambaNova)"},
    "cerebras": {"url": "https://api.cerebras.ai/v1/chat/completions",
                 "key_env": "CEREBRAS_API_KEY", "model": "qwen-3.8-27b",
                 "label": "Qwen-3.8-27B (Cerebras)"},
}[args.gen2_provider]
if args.gen2_model:
    GEN2["model"] = args.gen2_model

OUT = Path(args.out_dir or "results")
STORE = Path(args.store).resolve()

assert os.environ.get("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY missing"
if STAGES & {"xllm", "judge"}:
    assert os.environ.get(GEN2["key_env"]), f"{GEN2['key_env']} missing"

# ---- point the retriever at the requested store BEFORE it is instantiated ----
import retrieval.hybrid as H                                    # noqa: E402
H.VECTORSTORE_DIR = STORE

# ---- stub the web searcher (same rationale as the repo runners) ----
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
from evaluation.metrics import (                                 # noqa: E402
    bertscore, bertscore_lite, entity_recall, faithfulness, answer_relevancy,
)
from config.settings import SPECIES_CATALOG, RERANK_TOP_K        # noqa: E402
from scipy.stats import ttest_rel, wilcoxon, pearsonr, spearmanr  # noqa: E402
import urllib.request, urllib.error  # noqa: E402
from run_multi_llm_bench import call_deepseek, build_prompt  # noqa: E402
from generation.grounded_generator import SYSTEM_PROMPT          # noqa: E402
from run_llm_judge import JUDGE_PROMPT, _parse_score             # noqa: E402

GEN2_MODEL, GEN2_LABEL = GEN2["model"], GEN2["label"]
_G2_URL = GEN2["url"]
_G2_HEADERS = {"Authorization": f"Bearer {os.environ.get(GEN2['key_env'], '')}",
               "Content-Type": "application/json",
               "User-Agent": "sirca-rag-eval/1.0",
               "Accept": "application/json"}


def _g2_post(payload, timeout=120, max_retries=6):
    """POST to the second generator, retrying 429/5xx and 402.

    402 is transient here: SambaNova's edge nodes serve a stale zero balance
    for a while after credits are topped up, so an immediate retry succeeds
    where the first call fails with PAYMENT_METHOD_REQUIRED.
    """
    body = json.dumps(payload).encode()
    delay = 3.0
    last = None
    for _ in range(max_retries):
        req = urllib.request.Request(_G2_URL, data=body, headers=_G2_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (402, 429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 45)
                continue
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last = e
            time.sleep(delay)
            delay = min(delay * 2, 45)
            continue
    raise last

QUERY_TEMPLATES = [
    "What phytochemical compounds are reported for {sp}?",
    "Describe the pharmacological activities of {sp}.",
    "How does {sp} compare to related Andean medicinal species in traditional use?",
    "What antioxidant or anti-inflammatory effects have been documented in {sp}?",
    "Which alkaloids or flavonoids appear in {sp} extracts?",
]


def call_gen2(prompt, system=SYSTEM_PROMPT, max_tokens=2000):
    """run_multi_llm_bench.call_cerebras, with provider/model configurable.

    Uses the plain OpenAI field names (max_tokens) rather than Cerebras's
    max_completion_tokens/reasoning_effort, which SambaNova does not document.
    """
    payload = {"model": GEN2_MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": max_tokens,
               "top_p": 1, "stream": False}
    msg = _g2_post(payload)["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning") or "").strip()


def call_gen2_judge(context, answer, max_tokens=200):
    """run_llm_judge.call_cerebras_judge, with provider/model configurable."""
    prompt = (JUDGE_PROMPT.replace("{context}", context[:6000])
                          .replace("{answer}", answer[:2000]))
    payload = {"model": GEN2_MODEL,
               "messages": [{"role": "system",
                             "content": "You are a strict evaluator. "
                                        "Respond only with the requested JSON."},
                            {"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": max_tokens,
               "top_p": 1, "stream": False}
    msg = _g2_post(payload)["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning") or "").strip()


def _patch_deepseek_budget():
    """Escalate max_tokens on empty content instead of retrying identically.

    deepseek-v4-flash returns a `reasoning_content` field that is billed
    against max_tokens. When the reasoning fills the whole budget the reply
    comes back with finish_reason='length' and an EMPTY `content`, and the
    repo's generator raises after 5 attempts. Those retries cannot help: at
    TEMPERATURE=0 the call is deterministic, so the same prompt overflows the
    same way every time. Doubling the budget does fix it (verified: three
    queries that returned empty at 1500 tokens all answered at 4000, and the
    reasoning shrank from ~6.6k to ~0.8k characters once it was no longer
    being truncated mid-loop). Raising the ceiling is therefore the actual
    remedy for the blank answers the paper attributes to a missing retry.
    """
    import generation.grounded_generator as G

    def _deepseek_escalating(self, prompt, max_tokens):
        from openai import OpenAI
        client = OpenAI(api_key=G.DEEPSEEK_API_KEY, base_url=G.DEEPSEEK_BASE_URL)
        budgets = [max_tokens, max_tokens * 2, max_tokens * 4]
        last = None
        for budget in budgets:
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=G.DEEPSEEK_MODEL,
                        messages=[{"role": "system", "content": G.SYSTEM_PROMPT},
                                  {"role": "user", "content": prompt}],
                        max_tokens=budget, temperature=G.TEMPERATURE, top_p=0.9)
                except Exception as e:                       # transport/HTTP
                    last = e
                    time.sleep(2.0 * (attempt + 1))
                    continue
                ans = (resp.choices[0].message.content or "").strip()
                if ans:
                    return G.GenerationResult(
                        answer=ans,
                        citations_used=G._extract_citation_indices(ans),
                        model=G.DEEPSEEK_MODEL,
                        tokens_generated=resp.usage.completion_tokens,
                    )
                last = RuntimeError(f"empty content at max_tokens={budget} "
                                    f"(finish={resp.choices[0].finish_reason})")
                break        # deterministic at T=0: go straight to a bigger budget
        raise RuntimeError(f"DeepSeek empty after escalating to "
                           f"{budgets[-1]} tokens: {last}")

    G.GroundedGenerator._deepseek_generate = _deepseek_escalating
    print("  [patch] deepseek budget escalation active "
          "(max_tokens x1/x2/x4 on empty content)", flush=True)


def build_new_species_cases(n=30, seed=42):
    """run_table2_extended.build_new_species_cases, reading the patched store."""
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


def write(name, obj):
    p = OUT / f"{name}_{args.tag}.json"
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> wrote {p.name}", flush=True)


def read_back(name):
    p = OUT / f"{name}_{args.tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


METRIC_KEYS = ("bertscore_f1", "semantic_similarity", "entity_recall",
               "faithfulness", "answer_relevancy")


def stage_xllm():
    print(f"\n[xllm] DeepSeek vs {GEN2_LABEL} [{GEN2_MODEL}], {len(BENCHMARK_SET)} queries",
          flush=True)
    retriever = H.HybridRetriever()
    retriever.load()
    per_llm = {n: {"answers": [], "contexts": [], "queries": [], "references": []}
               for n in ("deepseek", "gen2")}
    records = []
    t0 = time.time()
    for i, tc in enumerate(BENCHMARK_SET, 1):
        rc = retriever.retrieve_with_context(tc.query, top_k=RERANK_TOP_K)
        ctx, cits = rc["context"], rc["citations"]
        prompt = build_prompt(tc.query, ctx, cits)
        rec = {"query": tc.query, "reference": tc.reference_answer,
               "category": tc.category, "species": tc.relevant_species,
               "context": ctx, "answers": {}}
        for name, fn in (("deepseek", call_deepseek), ("gen2", call_gen2)):
            try:
                ans = fn(prompt)
            except Exception as e:
                ans = f"[ERROR] {e}"
            time.sleep(1.0 if name == "gen2" else 0.3)
            rec["answers"][name] = ans
            per_llm[name]["answers"].append(ans)
            per_llm[name]["contexts"].append(ctx)
            per_llm[name]["queries"].append(tc.query)
            per_llm[name]["references"].append(tc.reference_answer)
        records.append(rec)
        if i % 5 == 0:
            print(f"  [{i}/{len(BENCHMARK_SET)}] {time.time()-t0:.0f}s", flush=True)

    print("  errors: " + str({n: sum(1 for a in per_llm[n]["answers"]
                                     if a.startswith("[ERROR"))
                              for n in per_llm}), flush=True)
    write("multi_llm_answers", records)

    metrics = {}
    for name in ("deepseek", "gen2"):
        d = per_llm[name]
        bs = bertscore(d["answers"], d["references"], lang="en")
        sim = bertscore_lite(d["answers"], d["references"])
        er = entity_recall(d["answers"], d["references"], d["contexts"])
        ff = faithfulness(d["answers"], d["contexts"])
        ar = answer_relevancy(d["queries"], d["answers"])
        metrics[name] = {
            "bertscore_f1": bs.score, "semantic_similarity": sim.score,
            "entity_recall": er.score, "faithfulness": ff.score,
            "answer_relevancy": ar.score,
            "per_query": {
                "bertscore_f1": bs.details.get("per_sample_f1", []),
                "semantic_similarity": sim.details.get("per_sample", []),
                "entity_recall": er.details.get("per_sample", []),
                "faithfulness": ff.details.get("per_sample", []),
                "answer_relevancy": ar.details.get("per_sample", []),
            },
        }
        print(f"  {name}: " + "  ".join(f"{k}={metrics[name][k]:.4f}"
                                        for k in METRIC_KEYS), flush=True)

    tt = {"_gen2_model": GEN2_MODEL, "_gen2_label": GEN2_LABEL}
    for k in METRIC_KEYS:
        a = metrics["deepseek"]["per_query"][k]
        b = metrics["gen2"]["per_query"][k]
        stat, p = ttest_rel(a, b)
        tt[k] = {"deepseek": metrics["deepseek"][k],
                 "gen2": metrics["gen2"][k],
                 "delta": metrics["gen2"][k] - metrics["deepseek"][k],
                 "t": float(stat), "p": float(p)}
        print(f"  t-test {k:22s} delta={tt[k]['delta']:+.4f} p={p:.5f}", flush=True)

    write("multi_llm_metrics", metrics)
    write("multi_llm_ttests", tt)
    return records


def stage_judge(records):
    print(f"\n[judge] {GEN2_LABEL} grades DeepSeek answers, claim-level",
          flush=True)
    judged, scores = [], []
    for i, rec in enumerate(records, 1):
        ans = rec["answers"].get("deepseek", "")
        ctx = rec.get("context", "")
        try:
            raw = call_gen2_judge(ctx, ans)
            sc, total, grounded, reason = _parse_score(raw)
        except Exception as e:
            raw, sc, total, grounded, reason = f"ERR: {e}", -1.0, 0, 0, str(e)
        judged.append({"query": rec["query"], "category": rec.get("category", ""),
                       "judge_raw": raw, "judge_score": sc,
                       "judge_total_claims": total,
                       "judge_grounded_claims": grounded, "judge_reason": reason})
        scores.append(sc)
        if i % 10 == 0:
            print(f"  [{i}/{len(records)}]", flush=True)
        time.sleep(2.0)

    ff = faithfulness([r["answers"].get("deepseek", "") for r in records],
                      [r.get("context", "") for r in records])
    fid_pq = ff.details.get("per_sample", [])
    valid = [(s, f) for s, f in zip(scores, fid_pq) if s >= 0]
    corr = {"judge_model": GEN2_MODEL, "judge_label": GEN2_LABEL, "n_valid": len(valid),
            "n_scored": len(scores)}
    if valid:
        s_arr = np.array([v[0] for v in valid], float)
        f_arr = np.array([v[1] for v in valid], float)
        pe, sp = pearsonr(s_arr, f_arr), spearmanr(s_arr, f_arr)
        corr.update({"pearson_r": float(pe.statistic), "pearson_p": float(pe.pvalue),
                     "spearman_r": float(sp.statistic),
                     "spearman_p": float(sp.pvalue),
                     "judge_mean": float(s_arr.mean()),
                     "fidelity_mean": float(f_arr.mean()),
                     "n_perfect": int((s_arr == 1.0).sum())})
        print(f"  n={len(valid)} r={corr['pearson_r']:.3f} p={corr['pearson_p']:.4f} "
              f"| judge={corr['judge_mean']:.3f} fidelity={corr['fidelity_mean']:.3f} "
              f"| perfect={corr['n_perfect']}/{len(valid)}", flush=True)
    write("llm_judge_results", {"judged": judged, "correlation": corr})


def stage_fidelity():
    extended = BENCHMARK_SET + build_new_species_cases(30, seed=42)
    print(f"\n[fidelity] full vs no_reranker, DeepSeek, n={len(extended)}", flush=True)
    _patch_deepseek_budget()
    fid = {}
    for cfg_name in ("full", "no_reranker"):
        cfg = next(c for c in ABLATION_CONFIGS if c.name == cfg_name)
        agent, restore = _build_agent(cfg, backend="deepseek")
        t1 = time.time()
        _, answers, _, contexts, _, _ = _run_queries(agent, extended)
        restore()
        ff = faithfulness(answers, contexts)
        fid[cfg_name] = {"per_query": ff.details["per_sample"], "mean": ff.score,
                         "n_blank": sum(1 for a in answers if not a.strip())}
        print(f"  {cfg_name}: Fidelity={ff.score:.4f} "
              f"blank={fid[cfg_name]['n_blank']} ({time.time()-t1:.0f}s)", flush=True)

    a, b = fid["full"]["per_query"], fid["no_reranker"]["per_query"]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    nz = sum(1 for x, y in zip(a, b) if x != y)
    _, p2 = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    _, p1 = wilcoxon(a, b, zero_method="wilcox", alternative="greater")
    delta = fid["full"]["mean"] - fid["no_reranker"]["mean"]
    print(f"  full={fid['full']['mean']:.4f} "
          f"no_reranker={fid['no_reranker']['mean']:.4f} delta={delta:+.4f} "
          f"({delta/fid['full']['mean']*100:+.1f}% rel) p_two={p2:.5f} "
          f"p_one={p1:.5f} n_nonzero={nz}/{n}", flush=True)

    write("n5_fidelity_wilcoxon", {
        "store": str(STORE),
        "full_fidelity_per_query": fid["full"]["per_query"],
        "no_reranker_fidelity_per_query": fid["no_reranker"]["per_query"],
        "full_mean": fid["full"]["mean"],
        "no_reranker_mean": fid["no_reranker"]["mean"],
        "full_n_blank": fid["full"]["n_blank"],
        "no_reranker_n_blank": fid["no_reranker"]["n_blank"],
        "delta": delta, "delta_rel_pct": delta / fid["full"]["mean"] * 100,
        "wilcoxon_two_sided_p": float(p2), "wilcoxon_greater_p": float(p1),
        "n_nonzero": nz, "n": n,
    })


print(f"store = {STORE}", flush=True)
print(f"stages = {sorted(STAGES)} | gen2 = {GEN2_LABEL} [{GEN2_MODEL}]", flush=True)

recs = None
if "xllm" in STAGES:
    recs = stage_xllm()
if "judge" in STAGES:
    if recs is None:
        recs = read_back("multi_llm_answers")
        assert recs, "judge needs multi_llm_answers_<tag>.json — run xllm first"
    stage_judge(recs)
if "fidelity" in STAGES:
    stage_fidelity()
print("\ndone", flush=True)
