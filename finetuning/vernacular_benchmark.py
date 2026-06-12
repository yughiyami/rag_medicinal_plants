"""
Vernacular Benchmark: evaluate the system's resistance to common, market
and native-dialect plant names.

Compares:
  - Base e5-base (default SIRCA-RAG)
  - Fine-tuned e5-base (Plan A + vernacular adaptation)

Metric: Recall@10 over a query set where the species is identified
ONLY by a common/regional/market/dialect name.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass


# ------------------------------------------------------------------
# Vernacular benchmark: 20 queries using non-scientific names only
# ------------------------------------------------------------------
@dataclass
class VernacularCase:
    query: str           # uses ONLY common/regional/dialect name
    target_species: str  # ground-truth scientific name
    name_type: str       # "common" | "regional" | "market" | "quechua" | "aymara"


VERNACULAR_BENCHMARK = [
    # --- Common names (well-known) ---
    VernacularCase("Para qué sirve la uña de gato?",
                   "Uncaria tomentosa", "common"),
    VernacularCase("Cuáles son los beneficios de la maca?",
                   "Lepidium meyenii", "common"),
    VernacularCase("Propiedades medicinales del sangre de drago",
                   "Croton lechleri", "common"),
    VernacularCase("Usos del aguaymanto en medicina tradicional",
                   "Physalis peruviana", "common"),
    VernacularCase("Beneficios de la hoja de coca",
                   "Erythroxylum coca", "common"),

    # --- Regional / market names ---
    VernacularCase("Para qué sirve la muña en infusión?",
                   "Minthostachys mollis", "regional"),
    VernacularCase("Uso medicinal del yacón peruano",
                   "Smallanthus sonchifolius", "regional"),
    VernacularCase("Qué cura la huamanpinta?",
                   "Chuquiraga spinosa", "market"),
    VernacularCase("Beneficios de la chachacoma para el soroche",
                   "Senecio nutans", "regional"),
    VernacularCase("Para qué se usa la carqueja?",
                   "Baccharis genistelloides", "regional"),

    # --- Quechua / native names ---
    VernacularCase("Wamanripa propiedades curativas",
                   "Chuquiraga spinosa", "quechua"),
    VernacularCase("Qhuna como planta medicinal",
                   "Minthostachys mollis", "quechua"),
    VernacularCase("Qishwar usos andinos",
                   "Buddleja incana", "quechua"),
    VernacularCase("Yawar wayqu árbol medicinal",
                   "Croton lechleri", "quechua"),
    VernacularCase("Ayuk willku para la energía",
                   "Lepidium meyenii", "quechua"),

    # --- Short, ambiguous market queries ---
    VernacularCase("muña muña para qué sirve",
                   "Minthostachys mollis", "market"),
    VernacularCase("té de quenua respiratorio",
                   "Polylepis rugulosa", "market"),
    VernacularCase("propiedades del eucalipto",
                   "Eucalyptus globulus", "common"),
    VernacularCase("para qué sirve la manzanilla",
                   "Matricaria chamomilla", "common"),
    VernacularCase("usos del orégano andino",
                   "Origanum vulgare", "common"),
]


def evaluate_retriever(retriever, benchmark, top_k=10):
    """Run benchmark queries and compute Recall@k by name type."""
    results = {"by_type": {}, "overall": {}, "details": []}

    type_correct = {}
    type_total = {}
    overall_correct = 0
    overall_total = 0

    for case in benchmark:
        docs = retriever.retrieve(case.query, top_k=top_k * 3,
                                  rerank_top_k=top_k, use_rerank=True)
        # Check if any of the top-k docs mentions the target species
        found = False
        for d in docs:
            meta = d.get("metadata") or {}
            species_meta = set(s.lower() for s in (meta.get("species") or []))
            content = (d.get("content") or "").lower()
            if (case.target_species.lower() in species_meta or
                    case.target_species.lower() in content):
                found = True
                break

        nt = case.name_type
        type_total[nt] = type_total.get(nt, 0) + 1
        if found:
            type_correct[nt] = type_correct.get(nt, 0) + 1
            overall_correct += 1
        overall_total += 1

        results["details"].append({
            "query": case.query,
            "target": case.target_species,
            "type": nt,
            "hit@10": found,
        })

    for nt in type_total:
        results["by_type"][nt] = {
            "hit_rate": type_correct.get(nt, 0) / type_total[nt],
            "n": type_total[nt],
        }
    results["overall"] = {
        "hit_rate": overall_correct / overall_total,
        "n": overall_total,
    }
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base",
                        choices=["base", "finetuned"])
    args = parser.parse_args()

    # Patch the embedding model path in retriever
    from retrieval import hybrid as hybrid_mod
    from config import settings as cfg

    if args.model == "finetuned":
        ft_path = Path(__file__).parent / "model"
        cfg.EMBEDDING_MODEL = str(ft_path)
        print(f"Using FINE-TUNED model: {ft_path}")
    else:
        print(f"Using BASE model: {cfg.EMBEDDING_MODEL}")

    # Reload retriever with new config
    from importlib import reload
    reload(hybrid_mod)
    retriever = hybrid_mod.HybridRetriever()
    retriever.load()

    print(f"\nEvaluating {len(VERNACULAR_BENCHMARK)} vernacular queries...\n")
    results = evaluate_retriever(retriever, VERNACULAR_BENCHMARK)

    print("=== RESULTS BY NAME TYPE ===")
    for nt, r in sorted(results["by_type"].items()):
        print(f"  {nt:10s}: {r['hit_rate']:.2f} ({r['n']} queries)")
    print(f"\n  OVERALL Hit@10: {results['overall']['hit_rate']:.3f} "
          f"({results['overall']['n']} queries)")

    out_file = Path(__file__).parent / f"results_{args.model}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
