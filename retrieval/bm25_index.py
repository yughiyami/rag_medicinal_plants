"""
BM25 sparse retrieval component (C3).
Complements dense retrieval for exact term matching —
critical for biomedical terminology (compound names, species binomials).
"""
import json
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from config.settings import PROCESSED_DIR, VECTORSTORE_DIR


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class BM25Index:
    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._contents: list[str] = []
        self._metadata: list[dict] = []

    def build(self, contents: list[str], metadata: list[dict]):
        """Build BM25 index from chunk contents."""
        self._contents = contents
        self._metadata = metadata
        tokenized = [_tokenize(c) for c in contents]
        self._index = BM25Okapi(tokenized)
        print(f"[C3-BM25] Index built: {len(contents)} documents")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search BM25 index and return scored results."""
        if self._index is None:
            raise RuntimeError("BM25 index not built")

        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "score": float(scores[idx]),
                    "content": self._contents[idx],
                    "metadata": self._metadata[idx],
                    "index": int(idx),
                })
        return results

    def save(self, path: Optional[Path] = None):
        if path is None:
            path = VECTORSTORE_DIR
        path = Path(path)
        with open(path / "bm25_index.pkl", "wb") as f:
            pickle.dump({
                "index": self._index,
                "contents": self._contents,
                "metadata": self._metadata,
            }, f)
        print(f"[C3-BM25] Saved -> {path / 'bm25_index.pkl'}")

    def load(self, path: Optional[Path] = None):
        if path is None:
            path = VECTORSTORE_DIR
        path = Path(path)
        with open(path / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        self._index = data["index"]
        self._contents = data["contents"]
        self._metadata = data["metadata"]
        print(f"[C3-BM25] Loaded: {len(self._contents)} documents")

    @classmethod
    def from_chunks(cls, chunks_file: str = "chunks_full.json") -> "BM25Index":
        """Build BM25 index directly from chunks file."""
        chunks_path = PROCESSED_DIR / chunks_file
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        contents = [c["content"] for c in chunks]
        metadata = [c["metadata"] for c in chunks]
        idx = cls()
        idx.build(contents, metadata)
        return idx
