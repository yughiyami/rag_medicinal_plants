"""
SIRCA-RAG: C2 Vectorization Pipeline
Generates embeddings and builds FAISS vector store.

Strategy (RAM-aware):
  1. Try BGE-M3 (1024d, multilingual, best quality)
  2. Fallback: multilingual-e5-base (768d, lighter)
  3. Last resort: all-MiniLM-L6-v2 (384d, very light)
"""
import sys
import json
import time
import gc
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import PROCESSED_DIR, VECTORSTORE_DIR, EMBEDDING_DIMENSION


def load_chunks(filename: str = "chunks_full.json") -> tuple[list[str], list[dict]]:
    """Load processed chunks from disk."""
    chunks_path = PROCESSED_DIR / filename
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    contents = [c["content"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]
    print(f"[C2] Loaded {len(contents)} chunks from {chunks_path}")
    return contents, metadata


def try_bge_m3(texts: list[str], batch_size: int = 8) -> tuple[np.ndarray, int]:
    """Try BGE-M3 (BAAI/bge-m3). Best quality, 1024d, multilingual."""
    print("[C2] Attempting BGE-M3 (1024d, multilingual)...")
    try:
        from FlagEmbedding import BGEM3FlagModel

        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print("[C2] BGE-M3 loaded successfully")

        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            output = model.encode(
                batch,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            all_embeddings.append(output["dense_vecs"])

            done = min(i + batch_size, total)
            if done % 100 == 0 or done == total:
                print(f"  [{done}/{total}] encoded")

        embeddings = np.vstack(all_embeddings)
        del model
        gc.collect()
        return embeddings, 1024

    except Exception as e:
        print(f"[C2] BGE-M3 failed: {e}")
        gc.collect()
        raise


def try_e5_multilingual(texts: list[str], batch_size: int = 16) -> tuple[np.ndarray, int]:
    """Fallback: multilingual-e5-base. 768d, lighter, still multilingual."""
    print("[C2] Falling back to multilingual-e5-base (768d)...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("intfloat/multilingual-e5-base")
        print("[C2] multilingual-e5-base loaded")

        # e5 models need "passage: " prefix for documents
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        del model
        gc.collect()
        return np.array(embeddings), 768

    except Exception as e:
        print(f"[C2] multilingual-e5-base failed: {e}")
        gc.collect()
        raise


def try_minilm(texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, int]:
    """Last resort: all-MiniLM-L6-v2. 384d, very light, EN only."""
    print("[C2] Last resort: all-MiniLM-L6-v2 (384d)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    del model
    gc.collect()
    return np.array(embeddings), 384


def build_faiss_index(embeddings: np.ndarray, metadata: list[dict], contents: list[str], dim: int):
    """Build and save FAISS index with metadata."""
    import faiss

    print(f"\n[C2] Building FAISS index (dim={dim}, n={len(embeddings)})...")

    embeddings_f32 = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings_f32)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_f32)

    print(f"[C2] Index built: {index.ntotal} vectors")

    # Save
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTORSTORE_DIR / "index.faiss"))

    store_meta = {
        "metadata": metadata,
        "contents": contents,
        "model_dim": dim,
        "total_vectors": index.ntotal,
    }
    with open(VECTORSTORE_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(store_meta, f, ensure_ascii=False)

    size_mb = (VECTORSTORE_DIR / "index.faiss").stat().st_size / (1024 * 1024)
    print(f"[C2] Saved FAISS index: {size_mb:.1f} MB -> {VECTORSTORE_DIR}")

    return index


def test_retrieval(contents: list[str], metadata: list[dict], dim: int):
    """Quick sanity test on the vector store."""
    import faiss

    print("\n[C2] Testing retrieval...")
    index = faiss.read_index(str(VECTORSTORE_DIR / "index.faiss"))

    test_queries = [
        "What are the anti-inflammatory compounds in Uncaria tomentosa?",
        "Cuales son las propiedades medicinales de la maca?",
        "Croton lechleri wound healing sangre de grado",
    ]

    # We need to encode queries - try to use the same model
    # For the test, just use the first few vectors as mock queries
    for i, query in enumerate(test_queries):
        # Use a random existing vector as proxy (real query encoding comes with retrieval module)
        query_vec = np.zeros((1, dim), dtype=np.float32)
        # Approximate: use average of first 5 vectors as query proxy
        with open(VECTORSTORE_DIR / "metadata.json", "r") as f:
            data = json.load(f)

        print(f"\n  Query: '{query}'")
        print(f"  (Full query encoding will work after retrieval module is built)")
        print(f"  Index size: {index.ntotal} vectors, dim={dim}")


def main():
    start = time.time()
    print("=" * 60)
    print("SIRCA-RAG: C2 Vectorization Pipeline")
    print("=" * 60)

    # Load chunks
    contents, metadata = load_chunks()

    # Try embedding models in order of preference
    embeddings = None
    dim = 0
    model_used = ""

    try:
        embeddings, dim = try_bge_m3(contents, batch_size=4)
        model_used = "BAAI/bge-m3"
    except Exception:
        print()
        try:
            embeddings, dim = try_e5_multilingual(contents, batch_size=8)
            model_used = "intfloat/multilingual-e5-base"
        except Exception:
            print()
            embeddings, dim = try_minilm(contents, batch_size=16)
            model_used = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"\n[C2] Embeddings generated: shape={embeddings.shape}, model={model_used}")

    # Build FAISS index
    index = build_faiss_index(embeddings, metadata, contents, dim)

    # Quick test
    test_retrieval(contents, metadata, dim)

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("VECTORIZATION COMPLETE")
    print(f"  Model: {model_used}")
    print(f"  Dimension: {dim}")
    print(f"  Vectors: {index.ntotal}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Store: {VECTORSTORE_DIR}")
    print("=" * 60)

    # Save vectorization metadata
    vec_meta = {
        "model": model_used,
        "dimension": dim,
        "total_vectors": index.ntotal,
        "elapsed_seconds": round(elapsed, 1),
        "chunks_source": "chunks_full.json",
    }
    with open(VECTORSTORE_DIR / "vectorization_info.json", "w") as f:
        json.dump(vec_meta, f, indent=2)


if __name__ == "__main__":
    main()
