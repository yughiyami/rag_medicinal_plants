"""
SIRCA-RAG Evaluation Benchmark.
Runs the full pipeline on a curated test set and computes all metrics.
"""
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark_data import TestCase, BENCHMARK_SET
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

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'Score':>10}")
    print("-" * 47)
    for key, val in results.items():
        print(f"  {key:<33} {val:>10.4f}")
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
