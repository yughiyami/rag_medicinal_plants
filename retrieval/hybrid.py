"""
Hybrid Retrieval Engine (C3).
Combines BM25 sparse + BGE-M3 dense retrieval with cross-encoder reranking.

Architecture:
  Query -> [BM25 top-K] + [FAISS top-K] -> RRF fusion -> Cross-encoder rerank -> top-K'

This directly implements Equation 8 from the paper:
  score_hybrid(q,d) = alpha * score_dense(q,d) + (1-alpha) * score_sparse(q,d)

Extended with Reciprocal Rank Fusion (RRF) for more robust combination.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

from config.settings import (
    VECTORSTORE_DIR,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    HYBRID_ALPHA,
    CROSS_ENCODER_MODEL,
)
from retrieval.bm25_index import BM25Index


class HybridRetriever:
    """
    Hybrid sparse+dense retriever with cross-encoder reranking.
    Addresses the paper indio's limitation: they only use dense retrieval.
    """

    def __init__(
        self,
        vectorstore_path: Optional[Path] = None,
        alpha: float = HYBRID_ALPHA,
    ):
        self.alpha = alpha
        self._vs_path = vectorstore_path or VECTORSTORE_DIR
        self._faiss_index = None
        self._bm25_index = None
        self._contents: list[str] = []
        self._metadata: list[dict] = []
        self._embedder = None
        self._reranker = None

    def load(self):
        """Load all retrieval components from disk."""
        # FAISS dense index
        self._faiss_index = faiss.read_index(str(self._vs_path / "index.faiss"))
        with open(self._vs_path / "metadata.json", "r", encoding="utf-8") as f:
            store = json.load(f)
        self._contents = store["contents"]
        self._metadata = store["metadata"]

        self._embedding_model = "BAAI/bge-m3"
        info_path = self._vs_path / "vectorization_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            self._embedding_model = info.get("model", self._embedding_model)

        print(f"[C3] FAISS loaded: {self._faiss_index.ntotal} vectors (dim={self._faiss_index.d}, model={self._embedding_model})")

        # BM25 sparse index
        bm25_path = self._vs_path / "bm25_index.pkl"
        if bm25_path.exists():
            self._bm25_index = BM25Index()
            self._bm25_index.load(self._vs_path)
        else:
            print("[C3] Building BM25 index...")
            self._bm25_index = BM25Index()
            self._bm25_index.build(self._contents, self._metadata)
            self._bm25_index.save(self._vs_path)

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query with the same model used for indexing."""
        if self._embedder is None:
            model_name = getattr(self, "_embedding_model", "BAAI/bge-m3")
            if "e5" in model_name:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(model_name)
                self._embedder_type = "e5"
                print(f"[C3] e5 encoder loaded for queries: {model_name}")
            elif "MiniLM" in model_name or "minilm" in model_name.lower():
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(model_name)
                self._embedder_type = "st"
                print(f"[C3] SentenceTransformer encoder loaded: {model_name}")
            else:
                from FlagEmbedding import BGEM3FlagModel
                self._embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
                self._embedder_type = "bge"
                print("[C3] BGE-M3 encoder loaded for queries")

        if self._embedder_type == "e5":
            vec = self._embedder.encode([f"query: {query}"], normalize_embeddings=True)
            return vec[0]
        elif self._embedder_type == "st":
            vec = self._embedder.encode([query], normalize_embeddings=True)
            return vec[0]
        else:
            output = self._embedder.encode(
                [query],
                batch_size=1,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return output["dense_vecs"][0]

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        """FAISS dense retrieval."""
        query_vec = self._encode_query(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vec)
        scores, indices = self._faiss_index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "content": self._contents[idx],
                "metadata": self._metadata[idx],
                "index": int(idx),
                "source": "dense",
            })
        return results

    def _sparse_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 sparse retrieval."""
        results = self._bm25_index.search(query, top_k=top_k)
        for r in results:
            r["source"] = "sparse"
        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) — more robust than linear interpolation.
        RRF(d) = sum(1 / (k + rank_i(d))) for each ranking i.
        Weighted by alpha for dense vs (1-alpha) for sparse.
        """
        scores = {}
        doc_map = {}

        for rank, r in enumerate(dense_results):
            idx = r["index"]
            rrf_score = self.alpha / (k + rank + 1)
            scores[idx] = scores.get(idx, 0) + rrf_score
            doc_map[idx] = r

        for rank, r in enumerate(sparse_results):
            idx = r["index"]
            rrf_score = (1 - self.alpha) / (k + rank + 1)
            scores[idx] = scores.get(idx, 0) + rrf_score
            if idx not in doc_map:
                doc_map[idx] = r

        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = []
        for idx in sorted_indices:
            result = doc_map[idx].copy()
            result["rrf_score"] = scores[idx]
            result["source"] = "hybrid"
            fused.append(result)

        return fused

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Cross-encoder reranking for deep query-document interaction."""
        if not candidates:
            return []

        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(CROSS_ENCODER_MODEL)
            print(f"[C3] Cross-encoder loaded: {CROSS_ENCODER_MODEL}")

        pairs = [[query, c["content"]] for c in candidates]
        scores = self._reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        use_rerank: bool = True,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline:
        1. BM25 sparse search (top_k)
        2. FAISS dense search (top_k)
        3. RRF fusion
        4. Cross-encoder reranking (top rerank_top_k)
        """
        # Parallel retrieval
        dense_results = self._dense_search(query, top_k)
        sparse_results = self._sparse_search(query, top_k)

        # Fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        if use_rerank and fused:
            rerank_candidates = fused[:top_k]
            return self._rerank(query, rerank_candidates, rerank_top_k)

        return fused[:rerank_top_k]

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
    ) -> dict:
        """
        Retrieve and format for LLM generation context.
        Returns structured context with citation metadata.
        """
        results = self.retrieve(query, rerank_top_k=top_k)

        context_parts = []
        citations = []
        for i, r in enumerate(results):
            m = r["metadata"]
            citation = {
                "index": i + 1,
                "pmid": m.get("pmid", ""),
                "doi": m.get("doi", ""),
                "title": m.get("title", ""),
                "authors": m.get("authors", []),
                "year": m.get("year", ""),
                "source": m.get("source", ""),
                "species": m.get("species", []),
            }
            citations.append(citation)
            context_parts.append(f"[{i+1}] {r['content']}")

        return {
            "query": query,
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "num_results": len(results),
            "results": results,
        }
