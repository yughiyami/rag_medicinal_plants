"""
Regenerate the paper's five result figures from the revised experiment outputs.

Differences from docs/make_figures.py, driven by the review:
  - legends and value labels moved out of the plotting area (R1.2, R2.5);
  - abbreviations expanded in-figure (Sem. Sim., Answer Rel., Entity Cov.);
  - n and the statistical test named on every panel that shows a comparison;
  - significance asterisks replaced by explicit p-values, so the reader does
    not depend on a caption to decode them;
  - numbers taken from the re-run on the final index.

Writes into figures_rev1/ so the originals in docs/images/ are untouched.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
WS = REPO / "results"          # committed result artifacts
OUT = REPO / "paper" / "figures"  # where main_en_rev1.tex reads them from
OUT.mkdir(parents=True, exist_ok=True)

DS_BLUE, GEN_ORANGE = "#2c6fbb", "#e08a1e"
ACC, REF, WEB = "#c0392b", "#e0a030", "#2c6fbb"
FULL, NOR = "#c0392b", "#8fb0cc"
plt.rcParams.update({
    # Springer asks for >=300 dpi halftones and ~600 dpi line art; save at 600.
    "figure.dpi": 150, "savefig.dpi": 600, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "axes.axisbelow": True,
})

T = json.loads((WS / "multi_llm_ttests_uncapped.json").read_text(encoding="utf-8"))
F = json.loads((WS / "n5_fidelity_wilcoxon_uncapped.json").read_text(encoding="utf-8"))
AB = json.loads((WS / "ablation_uncapped.json").read_text(encoding="utf-8"))
T1 = json.loads((WS / "table1_uncapped.json").read_text(encoding="utf-8"))
NP = json.loads((WS / "norm_pool_uncapped.json").read_text(encoding="utf-8"))["summary"]


def _p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def fig_headline():
    labels = ["Context Recall@10", "MRR", "NDCG@10", "Fidelity (65/35)"]
    a = AB["aggregate"]["full"]
    vals = [a["context_recall"], a["mrr"], a["ndcg@10"], F["full_mean"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    bars = ax.barh(labels[::-1], vals[::-1], color=DS_BLUE, height=0.62)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("score (higher is better)")
    ax.set_title("Full pipeline on the 50-query bilingual benchmark;\n"
                 "Fidelity over n=80 (50 benchmark + 30 additional species)",
                 fontsize=9, loc="left")
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "headline_metrics.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation_fidelity():
    ALL = json.loads((WS / "fidelity_all_configs_uncapped.json").read_text(encoding="utf-8"))
    full, nor = F["full_mean"], F["no_reranker_mean"]
    d = (np.array(F["full_fidelity_per_query"])
         - np.array(F["no_reranker_fidelity_per_query"]))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.3),
                                  gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.32})
    names = ["full", "dense\nonly", "no\ncrag", "sparse\nonly", "no\nreranker"]
    vals = [full, ALL["dense_only"]["mean"], ALL["no_crag"]["mean"],
            ALL["sparse_only"]["mean"], nor]
    cols = [FULL] + [NOR] * 4
    bars = ax.bar(names, vals, color=cols, width=0.62)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Fidelity (65 % semantic / 35 % lexical)")
    # Panel titles would repeat the LaTeX caption; label the panels instead.
    ax.set_title("(a) mean Fidelity, n=80", fontsize=8.5, loc="left")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax2.axvline(0, color="0.55", lw=1.0, ls="--", label="zero")
    ax2.hist(d, bins=20, color=FULL, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax2.axvline(d.mean(), color="#15521a", lw=1.7, label=f"mean (+{d.mean():.3f})")
    ax2.set_xlabel("per-query difference (full − no reranker)")
    ax2.set_ylabel("queries")
    ax2.set_title("(b) full − no_reranker, per query", fontsize=8.5, loc="left")
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "ablation_fidelity.png", bbox_inches="tight")
    plt.close(fig)


def fig_crag_routing():
    fams = ["A_missing_species", "B_off_domain", "C_garbled"]
    labels = ["A. Species with no\nindexed data", "B. Off-domain", "C. Garbled strings"]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = np.arange(len(fams))
    wd = 0.36
    import matplotlib.patches as mp
    for j, arm in enumerate(("action_minmax", "action_sigmoid")):
        bottom = np.zeros(len(fams))
        hatch = "" if j == 0 else "//"
        for act, col in (("accept", ACC), ("refine", REF), ("web_search", WEB)):
            v = np.array([NP["probes_by_family"][f][arm][act] for f in fams], float)
            ax.bar(x + (j - 0.5) * wd, v, wd * 0.9, bottom=bottom, color=col,
                   hatch=hatch, edgecolor="white", linewidth=0.6)
            bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 12.4)
    ax.set_ylabel("out-of-distribution probes")
    n_probes = NP["probes"]["n"]
    acc_mm = NP["probes"]["action_minmax"]["accept"]
    acc_sig = NP["probes"]["action_sigmoid"]["accept"]
    web_b = NP["probes_by_family"]["B_off_domain"]["action_sigmoid"]["web_search"]
    n_b = sum(NP["probes_by_family"]["B_off_domain"]["action_sigmoid"].values())
    # The message (acc_mm vs acc_sig, off-domain web_b/n_b) lives in the LaTeX caption.
    assert (acc_mm, acc_sig, web_b, n_b, n_probes) == (20, 8, 10, 10, 27)
    h = [mp.Patch(color=c, label=l) for c, l in
         ((ACC, "accept"), (REF, "refine"), (WEB, "external search"))]
    h += [mp.Patch(facecolor="0.75", edgecolor="white", label="within-batch min–max"),
          mp.Patch(facecolor="0.75", edgecolor="white", hatch="//",
                   label="absolute transformation")]
    ax.legend(handles=h, frameon=False, fontsize=7.5, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "crag_routing.png", bbox_inches="tight")
    plt.close(fig)


def fig_cross_llm():
    keys = ["bertscore_f1", "semantic_similarity", "entity_recall",
            "answer_relevancy", "faithfulness"]
    names = ["BERTScore F1", "Semantic\nSimilarity", "Entity\nCoverage",
             "Answer\nRelevancy", "Fidelity"]
    ds = [T[k]["deepseek"] for k in keys]
    gm = [T[k]["gen2"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, ds, w, label="DeepSeek V4-Flash", color=DS_BLUE)
    ax.bar(x + w / 2, gm, w, label="Gemma-4-31B", color=GEN_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Generator swap with the pipeline held fixed: each wins on different metrics\n"
                 "n=50 paired queries, two-tailed paired t-test",
                 fontsize=9, loc="left")
    for i, k in enumerate(keys):
        top = max(ds[i], gm[i])
        ax.text(i, top + 0.035, _p(T[k]["p"]), ha="center", fontsize=7,
                color="#7a1010" if T[k]["p"] < 0.05 else "0.35")
        ax.text(i - w / 2, ds[i] / 2, f"{ds[i]:.3f}", ha="center", va="center",
                fontsize=7, color="white", rotation=90)
        ax.text(i + w / 2, gm[i] / 2, f"{gm[i]:.3f}", ha="center", va="center",
                fontsize=7, color="white", rotation=90)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(OUT / "cross_llm.png", bbox_inches="tight")
    plt.close(fig)


def fig_coverage():
    names = ["C. Precision", "C. Recall", "MRR", "NDCG@5", "NDCG@10"]
    ks = ["context_precision", "context_recall", "mrr", "ndcg@5", "ndcg@10"]
    a = [T1["row_12_human_verified"][k] for k in ks]
    b = [T1["row_30new_species"][k] for k in ks]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    x = np.arange(len(ks))
    w = 0.38
    ax.bar(x - w / 2, a, w, label="12 human-verified species (50 queries)", color=DS_BLUE)
    ax.bar(x + w / 2, b, w, label="30 additional species (30 queries)", color="#2e9e5b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Retrieval quality is preserved on species outside the verified subset\n"
                 "(all species are present in the index; this is not out-of-corpus generalisation)",
                 fontsize=9, loc="left")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(i - w / 2, u + 0.018, f"{u:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, v + 0.018, f"{v:.3f}", ha="center", fontsize=7)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(OUT / "coverage_generalization.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Camera-ready uses only these two plots (+ the TikZ architecture diagram);
    # headline, cross-LLM and coverage duplicated tables and were dropped (R2-5).
    fig_ablation_fidelity()
    fig_crag_routing()
    print("figures written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.name)
