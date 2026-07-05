"""
Multi-LLM benchmark runner (reviewer O9 + O7 leverage).

Runs the SIRCA-RAG pipeline over the 50-query benchmark twice — once with
DeepSeek V4 Flash and once with Cerebras Gemma-4-31B — saving per-query
answers, contexts, retrieved chunks, and metric arrays for each backend.
The two output files feed:
  - O9: cross-LLM Wilcoxon on Fidelity / BERTScore / Semantic Similarity
  - O7: Cerebras acts as an independent LLM judge on DeepSeek answers
        (see run_llm_judge.py which consumes the persisted answers).
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.benchmark_data import BENCHMARK_SET
from evaluation.metrics import (
    bertscore_lite,
    context_precision,
    context_recall,
    mrr,
    ndcg_at_k,
    entity_recall,
    faithfulness,
    answer_relevancy,
)
from retrieval.hybrid import HybridRetriever
from agent.query_classifier import classify_query
from generation.grounded_generator import SYSTEM_PROMPT, CONTEXT_TEMPLATE

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
CEREBRAS_KEY = os.environ.get("CEREBRAS_API_KEY", "")


def _load_env():
    global DEEPSEEK_KEY
    if DEEPSEEK_KEY:
        return
    for l in Path(".env").read_text().splitlines():
        if l.startswith("DEEPSEEK_API_KEY="):
            DEEPSEEK_KEY = l.split("=", 1)[1].strip()


def _post_json(url, headers, payload, timeout=120, max_retries=5):
    body = json.dumps(payload).encode()
    delay = 2.0
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise


def call_deepseek(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 800):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    r = _post_json("https://api.deepseek.com/v1/chat/completions", headers, payload)
    return r["choices"][0]["message"]["content"].strip()


def call_cerebras(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 2000):
    payload = {
        "model": "gemma-4-31b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
        "top_p": 1,
        "stream": False,
        "reasoning_effort": "low",
    }
    headers = {
        "Authorization": f"Bearer {CEREBRAS_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "sirca-rag-eval/1.0",
        "Accept": "application/json",
    }
    r = _post_json("https://api.cerebras.ai/v1/chat/completions", headers, payload)
    msg = r["choices"][0]["message"]
    content = msg.get("content")
    if not content:
        # Fall back to reasoning text if content is empty (rare edge case)
        content = msg.get("reasoning") or ""
    return content.strip()


def build_prompt(query: str, context_docs, citations):
    citation_lines = []
    for c in citations:
        parts = [f"[{c['index']}]"]
        if c.get("title"):
            parts.append(c["title"][:80])
        if c.get("authors"):
            parts.append(f"({', '.join(c['authors'][:3])})")
        if c.get("year"):
            parts.append(str(c["year"]))
        citation_lines.append(" | ".join(parts))
    return CONTEXT_TEMPLATE.format(
        context=context_docs,
        citations="\n".join(citation_lines),
        query=query,
    )


def _retrieve_context(retriever, query):
    # Replicate the agent's _node_retrieve EXACTLY so the LLM context (and hence
    # Fidelity) matches the primary ablation (Table 4): classifier alpha, then
    # retrieve_with_context(top_k=RERANK_TOP_K=10) using full chunk content.
    from config.settings import RERANK_TOP_K

    original_alpha = retriever.alpha
    cls = classify_query(query)
    if cls.alpha_override is not None:
        retriever.alpha = cls.alpha_override

    rc = retriever.retrieve_with_context(query, top_k=RERANK_TOP_K)
    context_str = rc["context"]
    citations = rc["citations"]
    # retrieval-metric docs (same call the agent stores as retrieval_results)
    docs = retriever.retrieve(query, top_k=30, rerank_top_k=RERANK_TOP_K)

    retriever.alpha = original_alpha
    return docs, docs[:RERANK_TOP_K], context_str, citations


def main():
    _load_env()
    print("=" * 70)
    print("Multi-LLM benchmark: DeepSeek V4 Flash vs Cerebras Gemma-4-31B")
    print("=" * 70)

    retriever = HybridRetriever()
    retriever.load()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    llm_calls = {"deepseek": call_deepseek, "cerebras": call_cerebras}
    labels = {"deepseek": "DeepSeek V4 Flash", "cerebras": "Cerebras Gemma-4-31B"}

    all_records = []
    per_llm = {name: {"answers": [], "contexts": [], "queries": [],
                       "references": [], "retrieved": [], "relevant": []}
               for name in llm_calls}

    t0 = time.time()
    for i, tc in enumerate(BENCHMARK_SET, 1):
        print(f"\n[{i:02d}/{len(BENCHMARK_SET)}] {tc.query[:70]}")
        docs, top5, context_str, citations = _retrieve_context(retriever, tc.query)

        pool_ids = [int(d.get("index", -1)) for d in docs]
        retrieved_ids = [int(d.get("index", -1)) for d in top5]

        relevant_pool_ids = set()
        for d, cid in zip(docs, pool_ids):
            meta = d.get("metadata") or {}
            species_meta = set(s.lower() for s in (meta.get("species") or []))
            text = (d.get("content") or "").lower()
            for sp in tc.relevant_species:
                if sp.lower() in species_meta or sp.lower() in text:
                    relevant_pool_ids.add(cid)
                    break

        id_to_idx = {cid: j for j, cid in enumerate(pool_ids)}
        for cid in retrieved_ids:
            if cid not in id_to_idx:
                id_to_idx[cid] = len(id_to_idx)
        retrieved_idx = [id_to_idx[cid] for cid in retrieved_ids]
        relevant_idx = set(id_to_idx[cid] for cid in relevant_pool_ids)
        if not relevant_idx and retrieved_idx:
            relevant_idx.add(retrieved_idx[0])

        prompt = build_prompt(tc.query, context_str, citations)

        record = {"query": tc.query, "reference": tc.reference_answer,
                  "category": tc.category, "species": tc.relevant_species,
                  "context": context_str, "answers": {}}

        for name, fn in llm_calls.items():
            try:
                ans = fn(prompt)
            except urllib.error.HTTPError as e:
                ans = f"[ERROR:{e.code}] {e.reason}"
            except Exception as e:
                ans = f"[ERROR] {e}"
            # Small inter-request pace to soften rate limits
            time.sleep(1.0 if name == "cerebras" else 0.3)
            record["answers"][name] = ans
            per_llm[name]["answers"].append(ans)
            per_llm[name]["contexts"].append(context_str)
            per_llm[name]["queries"].append(tc.query)
            per_llm[name]["references"].append(tc.reference_answer)
            per_llm[name]["retrieved"].append(retrieved_idx)
            per_llm[name]["relevant"].append(relevant_idx)
            preview = ans[:80].replace(chr(10), " ")
            print(("  %-9s: %s..." % (name, preview)).encode("ascii", "replace").decode("ascii"))

        all_records.append(record)

    elapsed = time.time() - t0
    print(f"\nBenchmark done in {elapsed:.1f}s")

    (out_dir / "multi_llm_answers.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("Metrics per LLM (retrieval-side is identical; generation metrics differ)")
    print("=" * 70)

    from evaluation.metrics import bertscore
    from scipy.stats import ttest_rel

    metrics_out = {}
    for name in llm_calls:
        d = per_llm[name]
        print(f"\n--- {labels[name]} ---")
        bs = bertscore(d["answers"], d["references"], lang="en")
        sim = bertscore_lite(d["answers"], d["references"])
        er = entity_recall(d["answers"], d["references"], d["contexts"])
        ff = faithfulness(d["answers"], d["contexts"])
        ar = answer_relevancy(d["queries"], d["answers"])
        metrics_out[name] = {
            "bertscore_f1": bs.score,
            "semantic_similarity": sim.score,
            "entity_recall": er.score,
            "faithfulness": ff.score,
            "answer_relevancy": ar.score,
            "per_query": {
                "bertscore_f1": bs.details.get("per_sample_f1", []),
                "semantic_similarity": sim.details.get("per_sample", []),
                "entity_recall": er.details.get("per_sample", []),
                "faithfulness": ff.details.get("per_sample", []),
                "answer_relevancy": ar.details.get("per_sample", []),
            },
        }
        for k in ("bertscore_f1", "semantic_similarity", "entity_recall",
                  "faithfulness", "answer_relevancy"):
            print(f"  {k:22s} = {metrics_out[name][k]:.4f}")

    (out_dir / "multi_llm_metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8"
    )

    # Paired t-tests DeepSeek vs Cerebras
    print("\n" + "=" * 70)
    print("Paired t-tests: DeepSeek V4-Flash vs Cerebras Gemma-4-31B (n=50)")
    print("=" * 70)
    print(f"{'metric':22s} {'DeepSeek':>10s} {'Gemma':>10s} {'delta':>8s} {'t':>8s} {'p':>10s}")
    tt = {}
    ds, ce = metrics_out["deepseek"], metrics_out["cerebras"]
    for k in ("bertscore_f1", "semantic_similarity", "entity_recall",
              "faithfulness", "answer_relevancy"):
        a = ds["per_query"][k]
        b = ce["per_query"][k]
        try:
            stat, p = ttest_rel(a, b)
        except Exception as e:
            stat, p = float("nan"), float("nan")
        tt[k] = {"deepseek": ds[k], "gemma": ce[k],
                 "delta": ce[k] - ds[k], "t": float(stat), "p": float(p)}
        print(f"{k:22s} {ds[k]:10.4f} {ce[k]:10.4f} {ce[k]-ds[k]:+8.4f} "
              f"{stat:8.3f} {p:10.5f}")

    (out_dir / "multi_llm_ttests.json").write_text(json.dumps(tt, indent=2), encoding="utf-8")
    print("\nSaved -> results/multi_llm_metrics.json + multi_llm_ttests.json")
    print("\n(DeepSeek column = authoritative headline generation numbers for line 159)")


if __name__ == "__main__":
    main()
