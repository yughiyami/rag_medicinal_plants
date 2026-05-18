"""
BGE-M3 embedding module (C2).
Produces dense + sparse representations for hybrid retrieval.
Supports 100+ languages with 8192 token context window.
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, VECTORSTORE_DIR


class BGEM3Embedder:
    """
    Wrapper for BAAI/bge-m3 that produces:
    - Dense embeddings (1024d) for semantic similarity
    - Sparse embeddings (lexical weights) for BM25-like retrieval
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name, use_fp16=self.use_fp16
            )
            print(f"[C2] Loaded {self.model_name}")
        return self._model

    def encode_dense(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Generate dense embeddings (1024d)."""
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"]

    def encode_sparse(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """Generate sparse (lexical weight) embeddings."""
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return output["lexical_weights"]

    def encode_hybrid(
        self, texts: list[str], batch_size: int = 16
    ) -> tuple[np.ndarray, list[dict]]:
        """Generate both dense and sparse embeddings in one pass."""
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"], output["lexical_weights"]

    def encode_query(self, query: str) -> tuple[np.ndarray, dict]:
        """Encode a single query for retrieval."""
        dense, sparse = self.encode_hybrid([query], batch_size=1)
        return dense[0], sparse[0]


class VectorStore:
    """
    FAISS-based vector store with metadata persistence.
    Uses IVF-Flat for moderate-scale corpora (<1M documents).
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self._index = None
        self._metadata: list[dict] = []
        self._chunk_contents: list[str] = []

    @property
    def index(self):
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(self.dimension)
            print(f"[C2] Created FAISS IndexFlatIP (dim={self.dimension})")
        return self._index

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
        contents: list[str],
    ):
        """Add vectors with metadata to the store."""
        import faiss

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self._metadata.extend(metadata)
        self._chunk_contents.extend(contents)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[dict]:
        """Search for top-k similar vectors."""
        import faiss

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "content": self._chunk_contents[idx],
                "metadata": self._metadata[idx],
            })
        return results

    def save(self, path: Optional[Path] = None):
        """Persist index and metadata to disk."""
        import faiss

        if path is None:
            path = VECTORSTORE_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {"metadata": self._metadata, "contents": self._chunk_contents},
                f,
                ensure_ascii=False,
            )
        print(f"[C2] Saved vector store ({self.index.ntotal} vectors) → {path}")

    def load(self, path: Optional[Path] = None):
        """Load index and metadata from disk."""
        import faiss

        if path is None:
            path = VECTORSTORE_DIR
        path = Path(path)

        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self._metadata = data["metadata"]
        self._chunk_contents = data["contents"]
        print(f"[C2] Loaded vector store ({self._index.ntotal} vectors) ← {path}")

    @property
    def size(self) -> int:
        return self.index.ntotal if self._index else 0
