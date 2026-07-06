"""
Generate documentation figures from the real experiment results.
Outputs PNGs into docs/images/ used by the project README.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
IMG = Path(__file__).resolve().parent / "images"
IMG.mkdir(parents=True, exist_ok=True)

# Palette
BLUE = "#2c6fbb"
ORANGE = "#e08a1e"
GREEN = "#2e9e5b"
RED = "#c0392b"
GRAY = "#8a8f98"
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def fig_ablation_fidelity():
    # From results/n5_fidelity_wilcoxon.json (final run, fully bug-fixed agent, DeepSeek, n=80).
    # Direction (full > no_reranker) held in 5/5 runs across the debugging session, but the
    # Wilcoxon p-value ranged from 0.00023 to 0.160 run-to-run; this final run is NOT significant.
    cfgs = ["full", "no_reranker"]
    fid = [0.554, 0.526]
    colors = [GREEN, RED]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(cfgs, fid, color=colors, width=0.5)
    ax.set_ylim(0.40, 0.60)
    ax.set_ylabel("Fidelity (hybrid 65/35)")
    ax.set_title("Reranker → Fidelity: paired Wilcoxon on n=80 queries\n"
                 "-5.0% relative when removing the cross-encoder, p=0.160 (not significant)")
    for b, v in zip(bars, fid):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(1, 0.531), xytext=(0, 0.549),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.text(0.5, 0.542, "-5.0%\np=0.160 (n.s.)", ha="center", va="center",
            color=RED, fontsize=11, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=RED, lw=0.8))
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    fig.savefig(IMG / "ablation_fidelity.png", bbox_inches="tight")
    plt.close(fig)


def fig_crag_routing():
    d = json.load(open(ROOT / "results" / "crag_stress_absolute.json", encoding="utf-8"))
    fams = ["A_missing_species", "B_off_domain", "C_garbled"]
    labels = ["A. Missing-species\n(9 catalogued, 0 chunks)",
              "B. Off-domain\n(10 non-ethnobotany)",
              "C. Garbled\n(8 adversarial)"]
    acc = [d["summary"][f]["accept"] for f in fams]
    ref = [d["summary"][f]["refine"] for f in fams]
    web = [d["summary"][f]["web_search"] for f in fams]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(fams))
    ax.bar(x, acc, label="accept", color=GREEN)
    ax.bar(x, ref, bottom=acc, label="refine", color=ORANGE)
    ax.bar(x, web, bottom=[a + r for a, r in zip(acc, ref)], label="web_search", color=RED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of queries")
    ax.set_title("CRAG corrective-branch validation (27 out-of-distribution probes)\n"
                 "off-domain queries route to web_search 10/10")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for i, (a, r, w) in enumerate(zip(acc, ref, web)):
        if a: ax.text(i, a / 2, str(a), ha="center", va="center", color="white", fontsize=9)
        if r: ax.text(i, a + r / 2, str(r), ha="center", va="center", color="white", fontsize=9)
        if w: ax.text(i, a + r + w / 2, str(w), ha="center", va="center", color="white", fontsize=9)
    plt.tight_layout()
    fig.savefig(IMG / "crag_routing.png", bbox_inches="tight")
    plt.close(fig)


def fig_cross_llm():
    # From results/multi_llm_metrics.json + results/multi_llm_ttests.json
    metrics = ["BERTScore F1", "Sem. Sim.", "Entity Recall", "Answer Rel.", "Fidelity"]
    deepseek = [0.840, 0.816, 0.408, 0.993, 0.507]
    gemma = [0.828, 0.755, 0.392, 0.930, 0.607]
    sig = [True, False, False, False, True]  # paired t-test, alpha=0.05
    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar([i - w / 2 for i in x], deepseek, w, label="DeepSeek V4-Flash", color=BLUE)
    ax.bar([i + w / 2 for i in x], gemma, w, label="Cerebras Gemma-4-31B", color=ORANGE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Cross-LLM robustness (same pipeline, 50 queries)\n"
                 "Gemma-4 wins only on Fidelity; DeepSeek wins only BERTScore F1")
    for i, s in enumerate(sig):
        if s:
            ax.text(i, max(deepseek[i], gemma[i]) + 0.03, "*", ha="center",
                    fontsize=16, color=RED, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    fig.savefig(IMG / "cross_llm.png", bbox_inches="tight")
    plt.close(fig)


def fig_coverage():
    metrics = ["C.Prec.", "C.Recall", "MRR", "NDCG@10"]
    s12 = [0.474, 0.548, 0.862, 0.821]
    s30 = [0.505, 0.550, 0.883, 0.879]
    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([i - w / 2 for i in x], s12, w, label="12 species (human-verified)", color=BLUE)
    ax.bar([i + w / 2 for i in x], s30, w, label="30 new species (generalisation)", color=GREEN)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Coverage generalisation — retrieval holds on unseen species\n"
                 "(42 unique species evaluated in total)")
    for i, (a, b) in enumerate(zip(s12, s30)):
        ax.text(i - w / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=8)
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    fig.savefig(IMG / "coverage_generalization.png", bbox_inches="tight")
    plt.close(fig)


def fig_headline():
    metrics = ["Recall@10", "MRR", "NDCG@10", "Fidelity"]
    vals = [0.548, 0.862, 0.821, 0.554]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bars = ax.barh(metrics[::-1], vals[::-1], color=[GREEN, BLUE, BLUE, ORANGE][::-1], height=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("SIRCA-RAG — full pipeline headline metrics (50-query benchmark)")
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(IMG / "headline_metrics.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_headline()
    fig_ablation_fidelity()
    fig_crag_routing()
    fig_cross_llm()
    fig_coverage()
    print("Figures written to", IMG)
    for p in sorted(IMG.glob("*.png")):
        print(" -", p.name)
