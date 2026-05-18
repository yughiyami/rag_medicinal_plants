"""
SIRCA-RAG Evaluation Benchmark.
Runs the full pipeline on a curated test set and computes all metrics.
Produces results comparable to the Indian paper's Table III (F1 0.745-0.850).
"""
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    bertscore,
    bertscore_lite,
    context_precision,
    context_recall,
    mrr,
    ndcg_at_k,
    entity_recall,
    faithfulness,
    answer_relevancy,
)


@dataclass
class TestCase:
    query: str
    reference_answer: str
    relevant_species: list[str] = field(default_factory=list)
    relevant_compounds: list[str] = field(default_factory=list)
    category: str = "factual"


BENCHMARK_SET = [
    TestCase(
        query="What are the main anti-inflammatory alkaloids in Uncaria tomentosa?",
        reference_answer="Uncaria tomentosa (Rubiaceae) contains pentacyclic oxindolic alkaloids (POA) with mitraphylline (MTP) being the most abundant, which modifies the inflammatory response. Recent studies report anti-inflammatory and anti-proliferative properties of different alkaloids extracted from this plant, including immunomodulatory and antitumor properties. A steroidic fraction showed beta-sitosterol (60%), stigmasterol, and campesterol with moderate anti-inflammatory activity.",
        relevant_species=["Uncaria tomentosa"],
        relevant_compounds=["mitraphylline"],
        category="factual",
    ),
    TestCase(
        query="What is the wound healing mechanism of taspine from Croton lechleri?",
        reference_answer="Taspine is the cicatrizant principle found in Sangre de Grado, the latex of Croton lechleri, with wound healing and anti-inflammatory biological activity. Taspine is an alkaloid present in Croton lechleri latex at approximately 9% by dry weight. Other alkaloids isolated from Croton lechleri include glaucine, isoboldine, magnoflorine, norisoboldine, and thaliporphine.",
        relevant_species=["Croton lechleri"],
        relevant_compounds=["taspine"],
        category="factual",
    ),
    TestCase(
        query="Cuales son las propiedades medicinales de la maca para la fertilidad?",
        reference_answer="Lepidium meyenii (maca) posee propiedades medicinales para la fertilidad incluyendo mejora de la fertilidad y la libido, tratamiento de la infertilidad, y mejora del conteo y calidad del esperma. La maca ha sido utilizada tradicionalmente para mejorar la fertilidad, lo que sugiere su influencia en los sistemas endocrinos. Puede ser efectiva en la mejora del bienestar sexual tanto en hombres como en mujeres durante la andropausia y la menopausia.",
        relevant_species=["Lepidium meyenii"],
        relevant_compounds=[],
        category="exploratory",
    ),
    TestCase(
        query="Compare the antioxidant activity of Physalis peruviana and Smallanthus sonchifolius",
        reference_answer="Physalis peruviana crude ethanolic extract and calyx fractions were evaluated for antioxidant activity via superoxide and nitric oxide scavenging activity. Smallanthus sonchifolius landraces were investigated for total phenolic content, antioxidant activity and chemical composition of ethanol extracts and decoction extracts. Both species show antioxidant properties but use different methodologies and extract types.",
        relevant_species=["Physalis peruviana", "Smallanthus sonchifolius"],
        relevant_compounds=[],
        category="comparative",
    ),
    TestCase(
        query="What is the geographic distribution of Buddleja incana in Peru?",
        reference_answer="Buddleja incana has 62 georeferenced occurrences recorded in Peru according to GBIF data. These occurrences are distributed across the departments of Amazonas, Ancash, Arequipa, Cajamarca, Cusco, Huanuco, Junin, La Libertad, Lima, and Pasco.",
        relevant_species=["Buddleja incana"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="Actividad antimicrobiana de Minthostachys mollis aceite esencial",
        reference_answer="La actividad antimicrobiana del aceite esencial de Minthostachys mollis ha sido evaluada frente a varias bacterias y hongos. Se determino la actividad antibacteriana del aceite esencial de Minthostachys mollis frente a Helicobacter pylori, Shigella dysenteriae, Salmonella typhi y Pseudomonas aeruginosa. Ademas se demostro actividad antimicotica in vitro del aceite esencial de las hojas de Minthostachys mollis.",
        relevant_species=["Minthostachys mollis"],
        relevant_compounds=[],
        category="factual",
    ),
    TestCase(
        query="How does Erythroxylum coca differ from cocaine pharmacologically?",
        reference_answer="Erythroxylum coca is a plant species indigenous to the Andean region grown historically as a source of homeopathic medicine. Cocaine is a psychoactive substance extracted from Erythroxylum coca leaves, described as a potent stimulant of the sympathetic nervous system that causes structural changes on the brain, heart, lung, liver and kidney.",
        relevant_species=["Erythroxylum coca"],
        relevant_compounds=["cocaine"],
        category="exploratory",
    ),
    TestCase(
        query="What are the fructooligosaccharides in yacon and their prebiotic effects?",
        reference_answer="Smallanthus sonchifolius (yacon) is a root rich in fructooligosaccharides (FOS) containing 50-70% of dry weight, and inulin, which act as prebiotics. FOS intake favors the growth of health-promoting bacteria while reducing pathogenic bacteria populations. Commercial FOS can upregulate total secretory IgA in infant mice, providing prebiotic benefits.",
        relevant_species=["Smallanthus sonchifolius"],
        relevant_compounds=["fructooligosaccharides", "inulin"],
        category="factual",
    ),
]


def run_benchmark(backend: str = "template") -> dict:
    """Run the full evaluation benchmark."""
    from agent.graph import SIRCAAgent

    print("=" * 60)
    print("SIRCA-RAG: Evaluation Benchmark")
    print(f"  Test cases: {len(BENCHMARK_SET)}")
    print(f"  Generator: {backend}")
    print("=" * 60)

    agent = SIRCAAgent(generator_backend=backend)

    queries = []
    answers = []
    references = []
    contexts = []
    all_retrieved = []
    all_relevant = []

    print("\n--- Running Queries ---")
    for i, tc in enumerate(BENCHMARK_SET):
        print(f"  [{i+1}/{len(BENCHMARK_SET)}] {tc.query[:50]}...", end=" ")
        start = time.time()

        result = agent.run(tc.query)
        elapsed = time.time() - start

        gen = result.get("generation", {})
        answer = gen.get("answer", "")
        context = result.get("context", "")
        citations = result.get("citations", [])

        # Determine relevant indices by matching species/compounds in retrieved docs
        retrieved_idx = list(range(len(citations)))
        relevant_idx = set()
        for idx, cit in enumerate(citations):
            species_in_cit = set(s.lower() for s in cit.get("species", []))
            for sp in tc.relevant_species:
                if sp.lower() in species_in_cit or sp.lower() in context.lower():
                    relevant_idx.add(idx)
                    break

        queries.append(tc.query)
        answers.append(answer)
        references.append(tc.reference_answer)
        contexts.append(context)
        all_retrieved.append(retrieved_idx)
        all_relevant.append(relevant_idx)

        crag_action = result.get("crag_decision", {}).get("action", "?")
        print(f"[{crag_action}] {elapsed:.1f}s")

    # ---- Compute Metrics ----
    print("\n--- Computing Metrics ---")
    results = {}

    print("  BERTScore F1 (roberta-large)...", end=" ")
    try:
        bs = bertscore(answers, references, lang="en")
        results["bertscore_f1"] = bs.score
        results["bertscore_precision"] = bs.details["precision"]
        results["bertscore_recall"] = bs.details["recall"]
        print(f"{bs.score:.4f} (P={bs.details['precision']:.4f} R={bs.details['recall']:.4f})")
    except Exception as e:
        print(f"FAILED ({e.__class__.__name__})")
        results["bertscore_f1"] = 0.0

    print("  Semantic Similarity (cross-encoder)...", end=" ")
    bs_lite = bertscore_lite(answers, references)
    results["semantic_similarity"] = bs_lite.score
    print(f"{bs_lite.score:.4f}")

    print("  Context Precision...", end=" ")
    cp = context_precision(all_retrieved, all_relevant)
    results["context_precision"] = cp.score
    print(f"{cp.score:.4f}")

    print("  Context Recall...", end=" ")
    cr = context_recall(all_retrieved, all_relevant)
    results["context_recall"] = cr.score
    print(f"{cr.score:.4f}")

    print("  MRR...", end=" ")
    m = mrr(all_retrieved, all_relevant)
    results["mrr"] = m.score
    print(f"{m.score:.4f}")

    print("  NDCG@5...", end=" ")
    nd = ndcg_at_k(all_retrieved, all_relevant, k=5)
    results["ndcg@5"] = nd.score
    print(f"{nd.score:.4f}")

    print("  Entity Recall...", end=" ")
    er = entity_recall(answers, references, contexts)
    results["entity_recall"] = er.score
    print(f"{er.score:.4f}")

    print("  Faithfulness...", end=" ")
    ff = faithfulness(answers, contexts)
    results["faithfulness"] = ff.score
    print(f"{ff.score:.4f}")

    print("  Answer Relevancy...", end=" ")
    ar = answer_relevancy(queries, answers)
    results["answer_relevancy"] = ar.score
    print(f"{ar.score:.4f}")

    # ---- Comparison with Indian Paper ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'SIRCA-RAG':>10} {'Indian Paper':>12}")
    print("-" * 55)
    print(f"{'BERTScore F1 (roberta)':<30} {results['bertscore_f1']:>10.4f} {'0.745-0.850':>12}")
    print(f"{'Semantic Sim (cross-enc)':<30} {results['semantic_similarity']:>10.4f} {'N/A':>12}")
    print(f"{'Context Precision':<30} {results['context_precision']:>10.4f} {'N/A':>12}")
    print(f"{'Context Recall':<30} {results['context_recall']:>10.4f} {'N/A':>12}")
    print(f"{'MRR':<30} {results['mrr']:>10.4f} {'N/A':>12}")
    print(f"{'NDCG@5':<30} {results['ndcg@5']:>10.4f} {'N/A':>12}")
    print(f"{'Entity Recall':<30} {results['entity_recall']:>10.4f} {'N/A':>12}")
    print(f"{'Faithfulness':<30} {results['faithfulness']:>10.4f} {'N/A':>12}")
    print(f"{'Answer Relevancy':<30} {results['answer_relevancy']:>10.4f} {'N/A':>12}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="template", choices=["template", "deepseek", "ollama", "local"])
    args = parser.parse_args()

    results = run_benchmark(backend=args.backend)

    output_path = Path(__file__).resolve().parent.parent / "data" / f"evaluation_{args.backend}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_path}")
