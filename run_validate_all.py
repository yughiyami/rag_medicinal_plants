"""
SIRCA-RAG: Environment & Pipeline Validation.
Checks all components work end-to-end.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = 0
FAIL = 0
BASE = Path(__file__).resolve().parent


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


print("=" * 60)
print("SIRCA-RAG: Full Validation")
print("=" * 60)

# ---- 1. Data files ----
print("\n--- 1. Data Files ---")
check("chunks_expanded.json", (BASE / "data/processed/chunks_expanded.json").exists())
check("index.faiss", (BASE / "data/vectorstore/index.faiss").exists())
check("metadata.json", (BASE / "data/vectorstore/metadata.json").exists())
check("bm25_index.pkl", (BASE / "data/vectorstore/bm25_index.pkl").exists())
check("vectorization_info.json", (BASE / "data/vectorstore/vectorization_info.json").exists())

# ---- 2. Vectorstore integrity ----
print("\n--- 2. Vectorstore ---")
try:
    import faiss
    import numpy as np
    from config.settings import VECTORSTORE_DIR

    info_path = VECTORSTORE_DIR / "vectorization_info.json"
    with open(info_path) as f:
        info = json.load(f)
    dim = info["dimension"]
    expected_model = info["model"]

    index = faiss.read_index(str(VECTORSTORE_DIR / "index.faiss"))
    check(f"FAISS: {index.ntotal} vectors, dim={index.d}", index.ntotal > 0)
    check(f"Dimension matches info ({dim})", index.d == dim)

    q = np.random.randn(1, dim).astype(np.float32)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, 3)
    check("FAISS search works", indices[0][0] >= 0)
except Exception as e:
    check("Vectorstore", False, str(e))

# ---- 3. BM25 ----
print("\n--- 3. BM25 Index ---")
try:
    from retrieval.bm25_index import BM25Index
    bm25 = BM25Index()
    bm25.load()
    results = bm25.search("Uncaria tomentosa alkaloids", top_k=3)
    check(f"BM25: {len(bm25._contents)} docs", len(bm25._contents) > 0)
    check("BM25 search returns results", len(results) > 0)
except Exception as e:
    check("BM25", False, str(e))

# ---- 4. Hybrid retriever ----
print("\n--- 4. Hybrid Retriever ---")
try:
    from retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever()
    retriever.load()
    check(f"Embedding model: {retriever._embedding_model}", True)
    result = retriever.retrieve("Uncaria tomentosa alkaloids", top_k=3)
    check(f"Hybrid retrieve: {len(result)} results", len(result) > 0)
    check("Results have metadata", all("metadata" in r for r in result))
except Exception as e:
    check("Hybrid retriever", False, str(e))

# ---- 5. Query classifier ----
print("\n--- 5. Query Classifier ---")
try:
    from agent.query_classifier import classify_query
    r1 = classify_query("IC50 of taspine Croton lechleri")
    r2 = classify_query("How do medicinal plants reduce inflammation?")
    r3 = classify_query("Compare Uncaria tomentosa vs Croton lechleri")
    check("factual", r1.category == "factual")
    check("exploratory", r2.category == "exploratory")
    check("comparative", r3.category == "comparative")
except Exception as e:
    check("Query classifier", False, str(e))

# ---- 6. CRAG evaluator ----
print("\n--- 6. CRAG Evaluator ---")
try:
    from agent.crag_evaluator import CRAGEvaluator
    evaluator = CRAGEvaluator()
    fake = [
        {"content": "Uncaria tomentosa contains alkaloids", "rerank_score": 0.8, "metadata": {}},
        {"content": "Random text", "rerank_score": 0.1, "metadata": {}},
    ]
    decision = evaluator.evaluate("alkaloids in cat's claw", fake)
    check(f"CRAG: {decision.action}", decision.action in ("accept", "refine", "web_search"))
except Exception as e:
    check("CRAG evaluator", False, str(e))

# ---- 7. Template generator ----
print("\n--- 7. Grounded Generator ---")
try:
    from generation.grounded_generator import GroundedGenerator
    gen = GroundedGenerator(backend="template")
    result = gen.generate(
        query="test",
        context="[1] Uncaria tomentosa contains mitraphylline.",
        citations=[{"index": 1, "title": "Test", "source": "pubmed"}],
    )
    check("Template generator works", len(result.answer) > 0)
except Exception as e:
    check("Generator", False, str(e))

# ---- 8. Full agent pipeline ----
print("\n--- 8. Agent Pipeline ---")
try:
    from agent.graph import SIRCAAgent
    agent = SIRCAAgent(retriever=retriever, generator_backend="template")
    result = agent.run("What alkaloids does Uncaria tomentosa contain?")
    trace_nodes = [t["node"] for t in result.get("trace", [])]
    check("Pipeline completes", "generate" in trace_nodes)
    check("All 4 stages", all(n in trace_nodes for n in ["classify", "retrieve", "evaluate", "generate"]))
    gen = result.get("generation", {})
    check("Answer generated", len(gen.get("answer", "")) > 20)
except Exception as e:
    check("Agent pipeline", False, str(e))

# ---- Summary ----
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"VALIDATION: {PASS}/{total} passed, {FAIL} failed")
print("ALL SYSTEMS GO" if FAIL == 0 else "FIX FAILURES BEFORE CONTINUING")
print("=" * 60)
