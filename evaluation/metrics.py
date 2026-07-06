"""
SIRCA-RAG Evaluation Metrics (Day 6).
Implements RAGAS-inspired metrics without external LLM dependency.

Metrics:
  1. BERTScore (P/R/F1) — semantic similarity between answer and reference
  2. Context Precision — fraction of retrieved docs that are relevant
  3. Context Recall — fraction of relevant docs that were retrieved
  4. MRR (Mean Reciprocal Rank) — position of first relevant result
  5. NDCG@K — normalized discounted cumulative gain
  6. Entity Recall — named entities from reference found in answer
  7. Faithfulness (proxy) — n-gram overlap between answer and context
  8. Answer Relevancy — semantic similarity between answer and query
"""
import re
import math
import numpy as np
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class EvalResult:
    metric: str
    score: float
    details: dict = field(default_factory=dict)


# ---- 1. BERTScore ----

def bertscore(predictions: list[str], references: list[str], lang: str = "en") -> EvalResult:
    """Compute BERTScore P/R/F1 using microsoft/deberta-xlarge-mnli."""
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions, references,
        lang=lang,
        model_type="roberta-large",
        verbose=False,
        batch_size=4,
    )
    return EvalResult(
        metric="bertscore",
        score=float(F1.mean()),
        details={
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
            "per_sample_f1": [float(f) for f in F1],
        },
    )


def bertscore_lite(predictions: list[str], references: list[str]) -> EvalResult:
    """Answer quality score via cross-encoder relevance (sigmoid-normalized)."""
    from sentence_transformers import CrossEncoder
    from config.settings import CROSS_ENCODER_MODEL

    ce = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = list(zip(predictions, references))
    scores = ce.predict(pairs)
    norm_scores = _sigmoid(np.array(scores))

    return EvalResult(
        metric="answer_quality",
        score=float(np.mean(norm_scores)),
        details={
            "per_sample": [float(s) for s in norm_scores],
            "model": CROSS_ENCODER_MODEL,
        },
    )


# ---- 2. Context Precision ----

def context_precision(
    retrieved_indices: list[list[int]],
    relevant_indices: list[set[int]],
) -> EvalResult:
    """
    Average Precision: fraction of retrieved docs at each rank that are relevant.
    Higher = relevant docs ranked higher.
    """
    avg_precisions = []
    for retrieved, relevant in zip(retrieved_indices, relevant_indices):
        if not relevant:
            avg_precisions.append(0.0)
            continue
        hits = 0
        precision_sum = 0.0
        for rank, idx in enumerate(retrieved, 1):
            if idx in relevant:
                hits += 1
                precision_sum += hits / rank
        ap = precision_sum / len(relevant) if relevant else 0.0
        avg_precisions.append(ap)

    return EvalResult(
        metric="context_precision",
        score=float(np.mean(avg_precisions)),
        details={"per_query": avg_precisions},
    )


# ---- 3. Context Recall ----

def context_recall(
    retrieved_indices: list[list[int]],
    relevant_indices: list[set[int]],
) -> EvalResult:
    """Fraction of relevant documents that appear in retrieved set."""
    recalls = []
    for retrieved, relevant in zip(retrieved_indices, relevant_indices):
        if not relevant:
            recalls.append(1.0)
            continue
        retrieved_set = set(retrieved)
        hit = len(relevant & retrieved_set)
        recalls.append(hit / len(relevant))

    return EvalResult(
        metric="context_recall",
        score=float(np.mean(recalls)),
        details={"per_query": recalls},
    )


# ---- 4. MRR ----

def mrr(
    retrieved_indices: list[list[int]],
    relevant_indices: list[set[int]],
) -> EvalResult:
    """Mean Reciprocal Rank — position of first relevant result."""
    rrs = []
    for retrieved, relevant in zip(retrieved_indices, relevant_indices):
        rr = 0.0
        for rank, idx in enumerate(retrieved, 1):
            if idx in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    return EvalResult(
        metric="mrr",
        score=float(np.mean(rrs)),
        details={"per_query": rrs},
    )


# ---- 5. NDCG@K ----

def ndcg_at_k(
    retrieved_indices: list[list[int]],
    relevant_indices: list[set[int]],
    k: int = 10,
) -> EvalResult:
    """Normalized Discounted Cumulative Gain at K."""
    ndcgs = []
    for retrieved, relevant in zip(retrieved_indices, relevant_indices):
        dcg = 0.0
        for rank, idx in enumerate(retrieved[:k], 1):
            if idx in relevant:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return EvalResult(
        metric=f"ndcg@{k}",
        score=float(np.mean(ndcgs)),
        details={"per_query": ndcgs, "k": k},
    )


# ---- 6. Entity Recall ----

ENTITY_PATTERN = re.compile(
    r"\b[A-Z][a-z]{2,} [a-z]{4,}\b"  # Species binomials (Uncaria tomentosa)
    r"|\b[A-Za-z]*(?:ine|ide|ol|one|ate|ene|oid|ase)\b"  # Compounds (case-insensitive)
    r"|\b(?:IC50|EC50|LD50|Ki|Kd|MIC|CMI)\b"  # Bioassay labels
    r"|\b\d+\.?\d*\s*(?:mg|ug|ng|uM|nM|mM|mg/mL|ug/mL|%)\b"  # Concentrations
    r"|\bNF-[kK][Bb]\b|\bTNF-?\s*alpha\b|\bIL-\d+\s*beta?\b"  # Signaling molecules
    r"|\bDPPH\b|\bABTS\b|\bFOS\b"  # Assay abbreviations
)

KNOWN_ENTITIES = [
    "mitraphylline", "isomitraphylline", "pteropodine", "rhynchophylline",
    "isorhynchophylline", "taspine", "macamides", "macaenes",
    "withanolide", "chlorogenic acid", "pulegone", "thymol", "mentone",
    "cocaine", "ecgonine", "catechin", "fructooligosaccharides",
    "inulin", "fibroblast", "collagen", "prebiotic",
]


def entity_recall(
    answers: list[str],
    references: list[str],
    contexts: list[str] | None = None,
) -> EvalResult:
    """
    Entity recall: fraction of scientific entities from the reference found in the answer.
    The `contexts` parameter is accepted for interface consistency with other
    metrics but is not used in scoring -- this measures whether the ANSWER
    itself surfaces the reference entities, not whether they merely appear
    somewhere in the retrieved context.
    Scoring: entities found in answer / entities in reference.
    """
    recalls = []
    for answer, reference in zip(answers, references):
        ref_entities = set(m.group().lower() for m in ENTITY_PATTERN.finditer(reference))
        for entity in KNOWN_ENTITIES:
            if entity.lower() in reference.lower():
                ref_entities.add(entity.lower())

        if not ref_entities:
            recalls.append(1.0)
            continue

        ans_lower = answer.lower()
        hits = sum(1 for e in ref_entities if e in ans_lower)
        recalls.append(hits / len(ref_entities))

    return EvalResult(
        metric="entity_recall",
        score=float(np.mean(recalls)),
        details={
            "per_sample": recalls,
            "avg_entities_per_ref": float(np.mean([
                len(_extract_entities(r)) for r in references
            ])),
        },
    )


def _extract_entities(text: str) -> set[str]:
    entities = set(m.group().lower() for m in ENTITY_PATTERN.finditer(text))
    for e in KNOWN_ENTITIES:
        if e.lower() in text.lower():
            entities.add(e.lower())
    return entities


# ---- 7. Faithfulness (hybrid: semantic + lexical) ----

def faithfulness(answers: list[str], contexts: list[str], n: int = 3) -> EvalResult:
    """
    Hybrid faithfulness: combines semantic (cross-encoder sentence-level) and
    lexical (content-word overlap) scores. Language-agnostic.

    Semantic component (weight 0.6): each answer sentence scored against full
    context via cross-encoder. Measures if the claim is supported by context.

    Lexical component (weight 0.4): content-word overlap after filtering
    stopwords and boilerplate. Measures exact term reuse.
    """
    from sentence_transformers import CrossEncoder
    from config.settings import CROSS_ENCODER_MODEL

    ce = CrossEncoder(CROSS_ENCODER_MODEL)

    semantic_weight = 0.65
    lexical_weight = 0.35

    scores = []
    semantic_scores = []
    lexical_scores = []

    for answer, context in zip(answers, contexts):
        ans_clean = re.sub(r'\[\d+\]', '', answer)
        ans_clean = re.sub(
            r'(?i)(based solely on the provided context,?\s*'
            r'|according to the provided context,?\s*'
            r'|seg[uú]n la informaci[oó]n proporcionada,?\s*'
            r'|basado en la informaci[oó]n proporcionada,?\s*'
            r'|based on the available evidence,?\s*)',
            '', ans_clean,
        )

        if not ans_clean.strip():
            scores.append(0.0)
            semantic_scores.append(0.0)
            lexical_scores.append(0.0)
            continue

        # --- Semantic: sentence-level cross-encoder ---
        sentences = _split_sentences(ans_clean)
        if sentences:
            pairs = [[s, context] for s in sentences]
            raw = ce.predict(pairs)
            sent_scores = _sigmoid(np.array(raw))
            sem_score = float(np.mean(sent_scores))
        else:
            sem_score = 0.0

        # --- Lexical: content-word overlap ---
        ctx_content = set(_get_content_words(context.lower()))
        ans_content = _get_content_words(ans_clean.lower())
        if ans_content:
            token_hits = sum(1 for w in ans_content if w in ctx_content)
            lex_score = token_hits / len(ans_content)
        else:
            lex_score = 1.0

        combined = semantic_weight * sem_score + lexical_weight * lex_score
        scores.append(combined)
        semantic_scores.append(sem_score)
        lexical_scores.append(lex_score)

    return EvalResult(
        metric="faithfulness",
        score=float(np.mean(scores)),
        details={
            "per_sample": scores,
            "semantic_avg": float(np.mean(semantic_scores)),
            "lexical_avg": float(np.mean(lexical_scores)),
        },
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling both EN and ES punctuation."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


# ---- 8. Answer Relevancy ----

def answer_relevancy(queries: list[str], answers: list[str]) -> EvalResult:
    """Semantic similarity between query and answer using cross-encoder."""
    from sentence_transformers import CrossEncoder
    from config.settings import CROSS_ENCODER_MODEL

    ce = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = [[q, a] for q, a in zip(queries, answers)]
    scores = ce.predict(pairs)
    norm = _sigmoid(np.array(scores))

    return EvalResult(
        metric="answer_relevancy",
        score=float(np.mean(norm)),
        details={"per_sample": [float(s) for s in norm]},
    )


# ---- Helpers ----

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "when", "where", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their",
    "el", "la", "los", "las", "un", "una", "de", "del", "en", "con",
    "por", "para", "que", "se", "es", "son", "fue", "como", "su", "sus",
    "al", "lo", "le", "les", "y", "o", "pero", "si", "no", "mas",
}


def _get_ngrams(text: str, n: int) -> set[tuple]:
    words = text.split()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def _get_content_words(text: str) -> list[str]:
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
