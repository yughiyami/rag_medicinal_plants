"""
SIRCA-RAG: Day 4 — LangGraph Agent + CRAG Test
Tests the full pipeline: classify -> retrieve -> evaluate -> generate
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.query_classifier import classify_query
from agent.graph import SIRCAAgent

print("=" * 60)
print("SIRCA-RAG: C3 Agent Pipeline Test")
print("=" * 60)

# Step 1: Test query classifier standalone
print("\n--- Step 1: Query Classifier ---")
test_queries = [
    ("What compounds does Uncaria tomentosa contain?", "factual"),
    ("How do Peruvian medicinal plants reduce inflammation?", "exploratory"),
    ("Compare antioxidant activity of maca vs cat's claw", "comparative"),
    ("IC50 of taspine Croton lechleri", "factual"),
    ("propiedades medicinales de la maca Lepidium meyenii", "exploratory"),
    ("Uncaria tomentosa vs Croton lechleri wound healing", "comparative"),
]

for query, expected in test_queries:
    result = classify_query(query)
    match = "OK" if result.category == expected else "MISS"
    print(f"  [{match}] {result.category} (conf={result.confidence:.2f}) | {query[:55]}")

# Step 2: Full agent pipeline (template generator for speed)
print("\n--- Step 2: Full Agent Pipeline ---")
agent = SIRCAAgent(generator_backend="template")

agent_queries = [
    "What are the anti-inflammatory alkaloids in Uncaria tomentosa?",
    "propiedades medicinales de la maca para la fertilidad",
    "taspine wound healing mechanism Croton lechleri sangre de grado",
    "Compare antioxidant activity between Physalis peruviana and Smallanthus sonchifolius",
]

for query in agent_queries:
    print(f"\nQUERY: {query}")
    print("-" * 50)

    start = time.time()
    result = agent.run(query)
    elapsed = time.time() - start

    classification = result.get("classification", {})
    crag = result.get("crag_decision", {})
    gen = result.get("generation", {})
    trace = result.get("trace", [])

    print(f"  Type: {classification.get('category', '?')} (conf={classification.get('confidence', 0):.2f})")
    print(f"  CRAG: {crag.get('action', '?')} | {crag.get('reason', '')}")
    print(f"  Answer: {gen.get('answer', '')[:150]}...")
    print(f"  Citations used: {gen.get('citations_used', [])}")
    print(f"  Trace: {' -> '.join(t['node'] for t in trace)}")
    print(f"  Time: {elapsed:.2f}s")

print("\n" + "=" * 60)
print("AGENT PIPELINE READY")
print("=" * 60)
