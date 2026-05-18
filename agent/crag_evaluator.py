"""
Corrective RAG (CRAG) Evaluator for SIRCA-RAG.
Implements the CRAG decision loop: given retrieved documents and a query,
decides whether to ACCEPT, REFINE, or trigger WEB_SEARCH.

Based on: Yan et al., "Corrective Retrieval Augmented Generation" (2024).

Decision logic:
  - ACCEPT: at least one document scores above relevance threshold
  - REFINE: documents partially relevant, rewrite query and re-retrieve
  - WEB_SEARCH: no relevant documents, fall back to external search
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CRAGDecision:
    action: str  # accept | refine | web_search
    confidence: float
    relevant_indices: list[int] = field(default_factory=list)
    reason: str = ""
    refined_query: str | None = None


RELEVANCE_ACCEPT = 0.35
RELEVANCE_PARTIAL = 0.15
MIN_RELEVANT_RATIO = 0.2


class CRAGEvaluator:
    """Evaluates retrieval quality and decides corrective action."""

    def __init__(self, cross_encoder=None):
        self._cross_encoder = cross_encoder

    def _load_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            from config.settings import CROSS_ENCODER_MODEL
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._cross_encoder

    def score_relevance(self, query: str, documents: list[str]) -> np.ndarray:
        """Score query-document relevance using cross-encoder."""
        if not documents:
            return np.array([])

        encoder = self._load_cross_encoder()
        pairs = [[query, doc] for doc in documents]
        scores = encoder.predict(pairs)
        return np.array(scores)

    def evaluate(
        self,
        query: str,
        results: list[dict],
        use_rerank_scores: bool = True,
    ) -> CRAGDecision:
        """
        Evaluate retrieval results and decide corrective action.

        If results already have rerank_score (from hybrid retriever),
        uses those directly. Otherwise computes fresh cross-encoder scores.
        """
        if not results:
            return CRAGDecision(
                action="web_search",
                confidence=1.0,
                reason="No documents retrieved",
            )

        if use_rerank_scores and all("rerank_score" in r for r in results):
            scores = np.array([r["rerank_score"] for r in results])
        else:
            documents = [r["content"] for r in results]
            scores = self.score_relevance(query, documents)

        norm_scores = _normalize_scores(scores)

        relevant = np.where(norm_scores >= RELEVANCE_ACCEPT)[0].tolist()
        partial = np.where(
            (norm_scores >= RELEVANCE_PARTIAL) & (norm_scores < RELEVANCE_ACCEPT)
        )[0].tolist()

        relevant_ratio = len(relevant) / len(results)

        if relevant_ratio >= MIN_RELEVANT_RATIO and len(relevant) >= 1:
            return CRAGDecision(
                action="accept",
                confidence=float(np.mean(norm_scores[relevant])),
                relevant_indices=relevant,
                reason=f"{len(relevant)}/{len(results)} docs above threshold",
            )

        if partial:
            refined = _refine_query(query, results, partial)
            return CRAGDecision(
                action="refine",
                confidence=float(np.mean(norm_scores[partial])),
                relevant_indices=partial,
                reason=f"{len(partial)} partially relevant, {len(relevant)} fully relevant",
                refined_query=refined,
            )

        return CRAGDecision(
            action="web_search",
            confidence=float(1.0 - np.max(norm_scores)) if len(norm_scores) > 0 else 1.0,
            reason=f"Max relevance score {float(np.max(norm_scores)):.3f} below threshold {RELEVANCE_PARTIAL}",
        )


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize cross-encoder scores to [0, 1] range."""
    if len(scores) == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-6:
        return np.full_like(scores, 0.5)
    return (scores - min_s) / (max_s - min_s)


def _refine_query(query: str, results: list[dict], partial_indices: list[int]) -> str:
    """
    Refine query by extracting key terms from partially relevant documents.
    Simple keyword expansion — no LLM needed.
    """
    expansion_terms = set()
    for idx in partial_indices[:3]:
        meta = results[idx].get("metadata", {})
        species = meta.get("species", [])
        for s in species:
            expansion_terms.add(s)

        title = meta.get("title", "")
        if title:
            for word in title.split()[:5]:
                if len(word) > 5 and word.isalpha():
                    expansion_terms.add(word.lower())

    if expansion_terms:
        additions = " ".join(list(expansion_terms)[:4])
        return f"{query} {additions}"
    return query
