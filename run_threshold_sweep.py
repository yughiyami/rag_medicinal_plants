"""
Accept-threshold sweep for the CRAG router (paper Table `tab:threshold_sweep`).

Recomputes the routing decision at a range of accept thresholds from the raw
cross-encoder logits already stored in results/norm_pool_uncapped.json, so the
sweep needs no retrieval pass and is exactly reproducible from committed data.

The refine threshold is held fixed at PARTIAL (0.30), matching the paper: only
the accept threshold moves. The decision function is imported from
run_norm_and_pool so the sweep cannot drift from the deployed rule.

Usage:
    python run_threshold_sweep.py [--pool results/norm_pool_uncapped.json]
                                  [--out results/threshold_sweep_uncapped.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_norm_and_pool import PARTIAL, MIN_RATIO, sigmoid

# Thresholds reported in the paper table, plus the finer grid used for the plateau claim.
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 .. 0.95


def decide_at(norm: np.ndarray, accept: float) -> str:
    """Deployed CRAG decision rule with a parameterised accept threshold."""
    if len(norm) == 0:
        return "web_search"
    rel = np.where(norm >= accept)[0]
    par = np.where((norm >= PARTIAL) & (norm < accept))[0]
    if len(rel) / len(norm) >= MIN_RATIO and len(rel) >= 1:
        return "accept"
    if len(par) > 0:
        return "refine"
    return "web_search"


def subsets(per_query: list[dict]) -> dict[str, list[dict]]:
    """Split the stored queries into the groups the paper reports."""
    return {
        "benchmark": [q for q in per_query if q.get("kind") == "benchmark"],
        "A_missing_species": [q for q in per_query if q.get("family") == "A_missing_species"],
        "B_off_domain": [q for q in per_query if q.get("family") == "B_off_domain"],
        "C_garbled": [q for q in per_query if q.get("family") == "C_garbled"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=Path("results/norm_pool_uncapped.json"),
                    help="normalization/pool run holding raw_rerank_scores")
    ap.add_argument("--out", type=Path, default=Path("results/threshold_sweep_uncapped.csv"))
    args = ap.parse_args()

    data = json.loads(args.pool.read_text(encoding="utf-8"))
    per_query = data["per_query"]
    groups = subsets(per_query)

    missing = [name for name, qs in groups.items() if not qs]
    if missing:
        raise SystemExit(f"no queries found for: {', '.join(missing)} — check --pool")

    # Pre-compute the sigmoid-normalised logit vector once per query.
    norms = {name: [sigmoid(q["raw_rerank_scores"]) for q in qs] for name, qs in groups.items()}

    rows = []
    for accept in THRESHOLDS:
        row = {"accept_threshold": accept}
        for name, vecs in norms.items():
            actions = [decide_at(v, accept) for v in vecs]
            row[f"accept_rate_{name}"] = round(actions.count("accept") / len(actions), 4)
            row[f"n_{name}"] = len(actions)
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")
    for row in rows:
        print(f"  accept={row['accept_threshold']:.2f}  "
              f"benchmark={row['accept_rate_benchmark']:.2%}  "
              f"A={row['accept_rate_A_missing_species']:.2%}  "
              f"B={row['accept_rate_B_off_domain']:.2%}  "
              f"C={row['accept_rate_C_garbled']:.2%}")


if __name__ == "__main__":
    main()
