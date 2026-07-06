"""
LLM-as-judge grounding validation (reviewer O7).

Consumes results/multi_llm_answers.json (produced by run_multi_llm_bench.py)
and asks Cerebras Gemma-4-31B to score each DeepSeek answer's grounding
against the retrieved context, independently of the reference answer.
Correlates the LLM-judge score with the automatic Fidelity metric to
validate the construct.

Rubric v2 (claim-level, continuous): the original v1 rubric asked for a
single holistic 0-4 integer score and produced 43/50 perfect scores (86%),
a ceiling effect that mechanically suppresses correlation with a continuous
metric regardless of whether the construct is valid. v2 instead asks the
judge to enumerate the answer's factual claims and grade each one, reporting
a continuous 0-100 score = grounded_claims / total_claims. This gives the
judge room to express partial grounding instead of collapsing to "perfect."
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from evaluation.metrics import faithfulness

CEREBRAS_KEY = os.environ.get("CEREBRAS_API_KEY", "")


JUDGE_PROMPT = """You are an evaluator checking whether an ANSWER's factual claims are grounded in the provided CONTEXT.

Steps:
1. Break the ANSWER down into its individual factual claims (a claim = one discrete assertion: a compound name, a mechanism, an effect, a species relationship, a numeric value, an absence-of-information statement, etc.). A short answer may have as few as 1-2 claims; a longer one may have 5+.
2. For each claim, judge independently whether it is directly supported by explicit statements in the CONTEXT (grounded) or not (fabricated, contradicted, off-topic, or drifting from what the CONTEXT actually says).
3. Count total_claims and grounded_claims.

Be strict: paraphrasing a scientific name, compound, concentration, or mechanism in a way that changes its meaning counts as NOT grounded, even if the general topic is right.

Respond in EXACTLY this JSON format on a single line and nothing else:
{"total_claims": <int>, "grounded_claims": <int>, "reason": "<one sentence citing the weakest claim, or 'all claims grounded'>"}

CONTEXT:
{context}

ANSWER:
{answer}
"""


def call_cerebras_judge(context: str, answer: str, max_tokens: int = 200):
    prompt = JUDGE_PROMPT.replace("{context}", context[:6000]).replace("{answer}", answer[:2000])
    payload = {
        "model": "gemma-4-31b",
        "messages": [
            {"role": "system", "content": "You are a strict evaluator. Respond only with the requested JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
        "top_p": 1,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {CEREBRAS_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "sirca-rag-eval/1.0",
        "Accept": "application/json",
    }
    body = json.dumps(payload).encode()
    delay = 3.0
    last_err = None
    for attempt in range(6):
        req = urllib.request.Request("https://api.cerebras.ai/v1/chat/completions",
                                      data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning") or ""
            return content.strip()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            # Transient network/DNS failures raise URLError, not HTTPError --
            # the branch above never caught these (same gap found and fixed
            # in run_multi_llm_bench.py's _post_json).
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
    raise last_err


def _parse_score(text: str) -> tuple[float, int, int, str]:
    """Returns (score_0_1, total_claims, grounded_claims, reason). score is -1 on parse failure."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = json.loads(line)
                total = int(d.get("total_claims", -1))
                grounded = int(d.get("grounded_claims", -1))
                if total > 0 and 0 <= grounded <= total:
                    return grounded / total, total, grounded, str(d.get("reason", ""))
            except Exception:
                continue
    return -1.0, 0, 0, text[:120]


def main():
    ans_path = Path("results/multi_llm_answers.json")
    if not ans_path.exists():
        print(f"ERROR: {ans_path} not found. Run run_multi_llm_bench.py first.")
        sys.exit(1)

    records = json.loads(ans_path.read_text(encoding="utf-8"))
    print(f"Records: {len(records)}")
    print("=" * 70)
    print("LLM-as-judge (Cerebras Gemma-4-31B) scoring DeepSeek V4 Flash answers")
    print("=" * 70)

    judged = []
    scores = []
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        ans_ds = rec["answers"].get("deepseek", "")
        ctx = rec.get("context", "")
        try:
            raw = call_cerebras_judge(ctx, ans_ds)
            score, total, grounded, reason = _parse_score(raw)
        except Exception as e:
            raw, score, total, grounded, reason = f"ERR: {e}", -1.0, 0, 0, str(e)
        judged.append({
            "query": rec["query"],
            "category": rec.get("category", ""),
            "answer": ans_ds,
            "judge_raw": raw,
            "judge_score": score,
            "judge_total_claims": total,
            "judge_grounded_claims": grounded,
            "judge_reason": reason,
        })
        scores.append(score)
        print(f"  [{i:02d}] score={score:.2f} ({grounded}/{total} claims)  reason={reason[:70]}")
        time.sleep(2.0)  # pace to respect Cerebras per-second limits

    print(f"\nJudged in {time.time()-t0:.1f}s")
    n_perfect = sum(1 for s in scores if s == 1.0)
    print(f"Perfect scores (1.0): {n_perfect}/{len(scores)} ({100*n_perfect/len(scores):.0f}%)")

    # Correlate with automatic Fidelity
    print("\n" + "=" * 70)
    print("Correlation: LLM-judge score vs automatic Fidelity")
    print("=" * 70)

    answers = [rec["answers"].get("deepseek", "") for rec in records]
    contexts = [rec.get("context", "") for rec in records]
    ff = faithfulness(answers, contexts)
    fid_per_query = ff.details.get("per_sample", [])
    valid = [(s, f) for s, f in zip(scores, fid_per_query) if s >= 0]
    if valid:
        s_arr = np.array([v[0] for v in valid], dtype=float)  # already 0-1
        f_arr = np.array([v[1] for v in valid], dtype=float)
        pear = pearsonr(s_arr, f_arr)
        spear = spearmanr(s_arr, f_arr)
        print(f"  n valid = {len(valid)}")
        print(f"  Pearson  r = {pear.statistic:.3f}  p = {pear.pvalue:.4f}")
        print(f"  Spearman r = {spear.statistic:.3f}  p = {spear.pvalue:.4f}")
        print(f"  Mean judge (0-1): {s_arr.mean():.3f}")
        print(f"  Mean Fidelity:    {f_arr.mean():.3f}")

    out = Path("results/llm_judge_results.json")
    out.write_text(json.dumps({
        "judged": judged,
        "correlation": {
            "n_valid": len(valid),
            "pearson_r": float(pear.statistic) if valid else None,
            "pearson_p": float(pear.pvalue) if valid else None,
            "spearman_r": float(spear.statistic) if valid else None,
            "spearman_p": float(spear.pvalue) if valid else None,
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
