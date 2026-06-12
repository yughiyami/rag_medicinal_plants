"""
Quick evaluation: Does the fine-tuned model bring vernacular names CLOSER
to scientific names in embedding space?

Compares base e5-base vs fine-tuned model on the vernacular benchmark.
Metric: mean cosine similarity (query=vernacular_name → doc=scientific_passage)
       + Top-1 hit rate on a closed set of 100 species.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetuning.vernacular_benchmark import VERNACULAR_BENCHMARK
from config.settings import SPECIES_CATALOG


def build_species_passages():
    passages = {}
    for sci, info in SPECIES_CATALOG.items():
        use = info.get("use", "")
        compounds = ", ".join(info.get("compounds", []))
        family = info.get("family", "")
        passages[sci] = (
            f"passage: {sci} ({family}) is used for {use}. "
            f"Active compounds include {compounds}."
        )
    return passages


def evaluate(model_name: str, model: SentenceTransformer, label: str):
    passages = build_species_passages()
    species_list = list(passages.keys())
    passage_texts = [passages[s] for s in species_list]

    # Encode all species passages once
    p_emb = model.encode(passage_texts, normalize_embeddings=True,
                         show_progress_bar=False)

    # Encode all queries
    queries = [f"query: {c.query}" for c in VERNACULAR_BENCHMARK]
    q_emb = model.encode(queries, normalize_embeddings=True,
                         show_progress_bar=False)

    # For each query: similarity to target species + top-1 species predicted
    target_sims = []
    top1_hits = 0
    top5_hits = 0
    by_type_hits = {}
    by_type_total = {}

    for i, case in enumerate(VERNACULAR_BENCHMARK):
        sims = (q_emb[i] @ p_emb.T)  # (100,)
        target_idx = species_list.index(case.target_species)
        target_sim = sims[target_idx]
        target_sims.append(float(target_sim))

        # Top-1 prediction
        top1 = species_list[np.argmax(sims)]
        top5 = [species_list[j] for j in np.argsort(sims)[-5:][::-1]]

        nt = case.name_type
        by_type_total[nt] = by_type_total.get(nt, 0) + 1
        if top1 == case.target_species:
            top1_hits += 1
            by_type_hits[nt] = by_type_hits.get(nt, 0) + 1
        if case.target_species in top5:
            top5_hits += 1

    mean_target_sim = float(np.mean(target_sims))
    top1_acc = top1_hits / len(VERNACULAR_BENCHMARK)
    top5_acc = top5_hits / len(VERNACULAR_BENCHMARK)

    print(f"\n=== {label} ({model_name}) ===")
    print(f"  Mean target similarity:  {mean_target_sim:.4f}")
    print(f"  Top-1 accuracy:          {top1_acc:.3f} "
          f"({top1_hits}/{len(VERNACULAR_BENCHMARK)})")
    print(f"  Top-5 accuracy:          {top5_acc:.3f} "
          f"({top5_hits}/{len(VERNACULAR_BENCHMARK)})")
    print(f"  By name type:")
    for nt in sorted(by_type_total):
        h = by_type_hits.get(nt, 0)
        print(f"    {nt:10s}: {h}/{by_type_total[nt]} = "
              f"{h/by_type_total[nt]:.2f}")

    return {
        "label": label,
        "model": model_name,
        "mean_target_sim": mean_target_sim,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "by_type": {
            nt: by_type_hits.get(nt, 0) / by_type_total[nt]
            for nt in by_type_total
        },
    }


def main():
    print("Loading BASE model: intfloat/multilingual-e5-base")
    base = SentenceTransformer("intfloat/multilingual-e5-base")
    base_results = evaluate("intfloat/multilingual-e5-base", base, "BASE")

    print("\nLoading FINE-TUNED model: finetuning/model")
    ft_path = Path(__file__).parent / "model"
    ft = SentenceTransformer(str(ft_path))
    ft_results = evaluate("sirca-e5-ethnobotany", ft, "FINE-TUNED")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON (FT vs BASE)")
    print("=" * 60)
    delta_sim = ft_results["mean_target_sim"] - base_results["mean_target_sim"]
    delta_top1 = ft_results["top1_acc"] - base_results["top1_acc"]
    delta_top5 = ft_results["top5_acc"] - base_results["top5_acc"]
    print(f"  Delta Mean target sim:  {delta_sim:+.4f} "
          f"({100 * delta_sim / base_results['mean_target_sim']:+.1f}%)")
    print(f"  Delta Top-1 acc:        {delta_top1:+.3f} "
          f"(absolute)")
    print(f"  Delta Top-5 acc:        {delta_top5:+.3f} "
          f"(absolute)")

    out = {"base": base_results, "finetuned": ft_results,
           "delta": {
               "mean_target_sim": delta_sim,
               "top1_acc": delta_top1,
               "top5_acc": delta_top5,
           }}
    out_file = Path(__file__).parent / "eval_similarity_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
