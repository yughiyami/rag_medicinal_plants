"""
Build BM25 index and test the full hybrid retrieval pipeline.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from config.settings import VECTORSTORE_DIR

print("=" * 60)
print("SIRCA-RAG: C3 Hybrid Retrieval Setup & Test")
print("=" * 60)

# Step 1: Build and save BM25 index
print("\n--- Step 1: Build BM25 Index ---")
bm25 = BM25Index.from_chunks("chunks_full.json")
bm25.save()

# Step 2: Load hybrid retriever
print("\n--- Step 2: Load Hybrid Retriever ---")
retriever = HybridRetriever()
retriever.load()

# Step 3: Test queries
test_queries = [
    # English biomedical query
    "anti-inflammatory alkaloids Uncaria tomentosa cat's claw",
    # Spanish query (tests bilingual BGE-M3)
    "propiedades medicinales de la maca Lepidium meyenii fertilidad",
    # Specific compound query (BM25 should excel here)
    "taspine wound healing Croton lechleri sangre de grado",
    # Cross-species pharmacological query
    "antioxidant activity Peruvian medicinal plants",
    # Geographic/ecological query
    "geographic distribution Physalis peruviana Peru",
]

print("\n--- Step 3: Hybrid Retrieval Tests ---")
for query in test_queries:
    print(f"\nQUERY: {query}")
    print("-" * 50)

    start = time.time()
    result = retriever.retrieve_with_context(query, top_k=3)
    elapsed = time.time() - start

    for i, citation in enumerate(result["citations"]):
        species = ", ".join(citation["species"]) if citation["species"] else "N/A"
        source = citation["source"]
        title = citation["title"][:60] if citation["title"] else "N/A"
        print(f"  [{i+1}] {source} | {species}")
        print(f"      {title}")

    print(f"  Time: {elapsed:.2f}s | Results: {result['num_results']}")

print("\n" + "=" * 60)
print("HYBRID RETRIEVAL READY")
print("=" * 60)
