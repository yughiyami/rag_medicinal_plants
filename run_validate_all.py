"""
SIRCA-RAG: Full Environment & Pipeline Validation
Checks every component built in Days 1-5 still works after cache migration.
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

print("=" * 60)
print("SIRCA-RAG: Full Validation")
print("=" * 60)

# ---- 1. File system ----
print("\n--- 1. Data Files ---")
base = Path(__file__).resolve().parent

check("raw/pubmed_full.json", (base / "data/raw/pubmed_full.json").exists())
check("raw/gbif_*.json", any((base / "data/raw").glob("gbif_*.json")))
check("raw/phytochemical_*.json", any((base / "data/raw").glob("phytochemical_*.json")))
check("raw/perunpdb_*.json", any((base / "data/raw").glob("perunpdb_*.json")))
check("raw/literature_bilingual_*.json", any((base / "data/raw").glob("literature_bilingual_*.json")))
check("processed/chunks_full.json", (base / "data/processed/chunks_full.json").exists())
check("vectorstore/index.faiss", (base / "data/vectorstore/index.faiss").exists())
check("vectorstore/metadata.json", (base / "data/vectorstore/metadata.json").exists())
check("vectorstore/bm25_index.pkl", (base / "data/vectorstore/bm25_index.pkl").exists())

# ---- 2. Config ----
print("\n--- 2. Config ---")
try:
    from config.settings import (
        VECTORSTORE_DIR, EMBEDDING_DIMENSION, CROSS_ENCODER_MODEL,
        HYBRID_ALPHA, RETRIEVAL_TOP_K, GENERATOR_MODEL,
    )
    check("config.settings imports", True)
    check("EMBEDDING_DIMENSION=1024", EMBEDDING_DIMENSION == 1024)
    check("VECTORSTORE_DIR exists", VECTORSTORE_DIR.exists())
except Exception as e:
    check("config.settings", False, str(e))

# ---- 3. FAISS index ----
print("\n--- 3. FAISS Index ---")
try:
    import faiss
    import numpy as np
    index = faiss.read_index(str(VECTORSTORE_DIR / "index.faiss"))
    check(f"FAISS load ({index.ntotal} vectors, dim={index.d})", index.ntotal > 0)
    check("FAISS dimension=1024", index.d == 1024)
    q = np.random.randn(1, 1024).astype(np.float32)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, 3)
    check("FAISS search works", indices[0][0] >= 0)
except Exception as e:
    check("FAISS", False, str(e))

# ---- 4. BM25 index ----
print("\n--- 4. BM25 Index ---")
try:
    from retrieval.bm25_index import BM25Index
    bm25 = BM25Index()
    bm25.load()
    results = bm25.search("Uncaria tomentosa alkaloids", top_k=3)
    check(f"BM25 load ({len(bm25._contents)} docs)", len(bm25._contents) > 0)
    check("BM25 search returns results", len(results) > 0)
except Exception as e:
    check("BM25", False, str(e))

# ---- 5. BGE-M3 embeddings (cache moved to D:) ----
print("\n--- 5. BGE-M3 Encoder ---")
try:
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    out = model.encode(["test query"], batch_size=1, return_dense=True,
                       return_sparse=False, return_colbert_vecs=False)
    vec = out["dense_vecs"][0]
    check(f"BGE-M3 loads from D:\\MLCache (dim={len(vec)})", len(vec) == 1024)
    del model
except Exception as e:
    check("BGE-M3", False, str(e))

# ---- 6. Cross-encoder ----
print("\n--- 6. Cross-Encoder ---")
try:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CROSS_ENCODER_MODEL)
    score = ce.predict([["test query", "test document"]])
    check(f"Cross-encoder loads ({CROSS_ENCODER_MODEL})", True)
    del ce
except Exception as e:
    check("Cross-encoder", False, str(e))

# ---- 7. Hybrid retriever ----
print("\n--- 7. Hybrid Retriever ---")
try:
    from retrieval.hybrid import HybridRetriever
    retriever = HybridRetriever()
    retriever.load()
    result = retriever.retrieve_with_context("Uncaria tomentosa alkaloids", top_k=3)
    check(f"Hybrid retrieve ({result['num_results']} results)", result["num_results"] > 0)
    check("Citations have metadata", len(result["citations"]) > 0 and "source" in result["citations"][0])
except Exception as e:
    check("Hybrid retriever", False, str(e))

# ---- 8. Query classifier ----
print("\n--- 8. Query Classifier ---")
try:
    from agent.query_classifier import classify_query
    r1 = classify_query("IC50 of taspine Croton lechleri")
    r2 = classify_query("How do medicinal plants reduce inflammation?")
    r3 = classify_query("Compare Uncaria tomentosa vs Croton lechleri")
    check("factual classification", r1.category == "factual")
    check("exploratory classification", r2.category == "exploratory")
    check("comparative classification", r3.category == "comparative")
except Exception as e:
    check("Query classifier", False, str(e))

# ---- 9. CRAG evaluator ----
print("\n--- 9. CRAG Evaluator ---")
try:
    from agent.crag_evaluator import CRAGEvaluator
    evaluator = CRAGEvaluator()
    fake_results = [
        {"content": "Uncaria tomentosa contains alkaloids", "rerank_score": 0.8, "metadata": {}},
        {"content": "Random unrelated text", "rerank_score": 0.1, "metadata": {}},
    ]
    decision = evaluator.evaluate("alkaloids in cat's claw", fake_results)
    check(f"CRAG decision: {decision.action}", decision.action in ("accept", "refine", "web_search"))
except Exception as e:
    check("CRAG evaluator", False, str(e))

# ---- 10. Web searcher ----
print("\n--- 10. Web Searcher ---")
try:
    from scraping.web_searcher import search_sync
    web_results = search_sync("Uncaria tomentosa alkaloids", max_results=2)
    check(f"Web search ({len(web_results)} results)", len(web_results) > 0)
    check("Web results have titles", all(r.title for r in web_results))
except Exception as e:
    check("Web searcher", False, str(e))

# ---- 11. LangGraph agent ----
print("\n--- 11. LangGraph Agent ---")
try:
    from agent.graph import SIRCAAgent
    agent = SIRCAAgent(generator_backend="template")
    result = agent.run("What alkaloids does Uncaria tomentosa contain?")
    trace_nodes = [t["node"] for t in result.get("trace", [])]
    check("Agent pipeline completes", "generate" in trace_nodes)
    check("Agent trace has classify", "classify" in trace_nodes)
    check("Agent trace has retrieve", "retrieve" in trace_nodes)
    check("Agent trace has evaluate", "evaluate" in trace_nodes)
    gen = result.get("generation", {})
    check("Agent generates answer", len(gen.get("answer", "")) > 20)
except Exception as e:
    check("LangGraph agent", False, str(e))

# ---- 12. Generation module ----
print("\n--- 12. Grounded Generator (template) ---")
try:
    from generation.grounded_generator import GroundedGenerator
    gen = GroundedGenerator(backend="template")
    result = gen.generate(
        query="test",
        context="[1] Uncaria tomentosa contains mitraphylline.",
        citations=[{"index": 1, "title": "Test", "source": "pubmed"}],
    )
    check("Template generator works", len(result.answer) > 0)
    check("Citations extracted", 1 in result.citations_used)
except Exception as e:
    check("Generator", False, str(e))

# ---- 13. Disk space ----
print("\n--- 13. Disk Space ---")
import shutil
c_total, c_used, c_free = shutil.disk_usage("C:\\")
d_total, d_used, d_free = shutil.disk_usage("D:\\")
check(f"C: free = {c_free//(1024**3)} GB", c_free > 2 * 1024**3, "Less than 2GB free!")
check(f"D: free = {d_free//(1024**3)} GB", d_free > 50 * 1024**3)

# HF cache symlink check
hf_path = Path.home() / ".cache" / "huggingface"
check("HF cache junction exists", hf_path.exists())
bge_path = hf_path / "hub" / "models--BAAI--bge-m3"
check("BGE-M3 model accessible via junction", bge_path.exists())

# ---- Summary ----
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"VALIDATION: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL SYSTEMS GO")
else:
    print("FIX FAILURES BEFORE CONTINUING")
print("=" * 60)
