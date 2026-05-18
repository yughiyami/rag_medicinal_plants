"""
SIRCA-RAG Main Pipeline
Orchestrates: Acquisition (C4) → Chunking (C1) → Embedding (C2) → Vector Store
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR
from ingestion.pubmed_client import acquire_all, save_raw
from ingestion.chunker import chunk_articles
from embeddings.bge_m3 import BGEM3Embedder, VectorStore


def run_acquisition(retmax: int = 50) -> Path:
    """Phase 1: Acquire articles from PubMed APIs."""
    print("=" * 60)
    print("PHASE 1: SCIENTIFIC ACQUISITION (C4)")
    print("=" * 60)
    articles = acquire_all(retmax_per_query=retmax)
    return save_raw(articles)


def run_chunking(raw_path: Path) -> Path:
    """Phase 2: Semantic chunking of acquired documents."""
    print("\n" + "=" * 60)
    print("PHASE 2: SEMANTIC CHUNKING (C1)")
    print("=" * 60)

    with open(raw_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"[C1] Processing {len(articles)} articles...")
    chunks = chunk_articles(articles)
    print(f"[C1] Generated {len(chunks)} chunks")

    # Save processed chunks
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = PROCESSED_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[C1] Saved chunks → {chunks_path}")

    return chunks_path


def run_embedding(chunks_path: Path) -> Path:
    """Phase 3: Generate BGE-M3 embeddings and build vector store."""
    print("\n" + "=" * 60)
    print("PHASE 3: EMBEDDING & INDEXING (C2)")
    print("=" * 60)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    contents = [c["content"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    print(f"[C2] Encoding {len(contents)} chunks with BGE-M3...")
    embedder = BGEM3Embedder()
    dense_embeddings = embedder.encode_dense(contents, batch_size=32)

    print(f"[C2] Building FAISS index...")
    store = VectorStore()
    store.add(dense_embeddings, metadata, contents)
    store.save()

    print(f"[C2] Vector store ready: {store.size} vectors indexed")
    return VECTORSTORE_DIR


def run_full_pipeline(retmax: int = 50):
    """Run the complete ingestion pipeline end-to-end."""
    print("\n🌿 SIRCA-RAG: Pipeline de Ingesta Completo")
    print("  Plantas Medicinales Peruanas")
    print("  BGE-M3 Multilingual Embeddings")
    print("=" * 60)

    raw_path = run_acquisition(retmax=retmax)
    chunks_path = run_chunking(raw_path)
    vectorstore_path = run_embedding(chunks_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Raw data: {raw_path}")
    print(f"  Chunks: {chunks_path}")
    print(f"  Vector store: {vectorstore_path}")
    print("=" * 60)


if __name__ == "__main__":
    retmax = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_full_pipeline(retmax=retmax)
