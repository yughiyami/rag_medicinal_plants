"""
Query Classifier for SIRCA-RAG (C3 Agent).
Classifies incoming queries into a category for routing/explainability purposes.

Categories:
  - factual: specific compound/species/property lookup
  - exploratory: broad research questions
  - comparative: cross-species or cross-compound analysis

Note: this classifier used to also assign a per-category retrieval alpha
override (factual->0.3, exploratory->0.7, comparative->0.5). A held-out
alpha sweep (run_alpha_sweep_heldout.py) showed that a flat alpha (0.6)
consistently outperforms per-category weighting, including when the
per-category values are properly tuned on held-out data rather than copied
from the original untested assumption. The alpha override was removed;
retrieval always uses the hybrid retriever's default alpha now.
"""
import re
from dataclasses import dataclass


@dataclass
class QueryClassification:
    category: str  # factual | exploratory | comparative
    confidence: float
    features: dict


COMPARATIVE_MARKERS = [
    r"\bcompar[eaing]",
    r"\bversus\b", r"\bvs\.?\b",
    r"\bdifference[s]?\b", r"\bdiferencia[s]?\b",
    r"\bbetter\b", r"\bmejor\b",
    r"\bsimilar\b",
    r"\bboth\b", r"\bambos\b",
    r"\brelation\b", r"\brelaci[oó]n\b",
]

FACTUAL_MARKERS = [
    r"\bwhat is\b", r"\bqu[eé] es\b",
    r"\bdefin[eition]", r"\bdefini[cr]",
    r"\bcontains?\b", r"\bcontiene\b",
    r"\bcompound[s]?\b", r"\bcompuesto[s]?\b",
    r"\bspecies\b", r"\bespecie[s]?\b",
    r"\bclassif", r"\btaxonom",
    r"\bstructure\b", r"\bestructura\b",
    r"\bIC50\b", r"\bEC50\b", r"\bLD50\b",
    r"\bSMILES\b", r"\bInChI\b",
    r"\bPMID\b", r"\bDOI\b",
]

EXPLORATORY_MARKERS = [
    r"\bhow\b", r"\bc[oó]mo\b",
    r"\bwhy\b", r"\bpor qu[eé]\b",
    r"\bmechanism\b", r"\bmecanismo\b",
    r"\bpathway\b", r"\bruta\b",
    r"\beffect[s]?\b", r"\befecto[s]?\b",
    r"\bpotential\b", r"\bpotencial\b",
    r"\bapplication\b", r"\baplicaci[oó]n\b",
    r"\btherapeutic\b", r"\bterap[eé]utic",
    r"\breview\b", r"\brevisi[oó]n\b",
    r"\bactivit", r"\bactividad",
    r"\bproperties\b", r"\bpropiedades\b",
]

SPECIES_PATTERN = re.compile(
    r"\b[A-Z][a-z]{2,} [a-z]{4,}\b"
)

NON_SPECIES_WORDS = {
    "Peruvian", "Indian", "African", "Chinese", "Brazilian",
    "South", "North", "Western", "Eastern", "Central",
    "Traditional", "Natural", "Medicinal", "Tropical",
}


def _count_matches(text: str, patterns: list[str]) -> int:
    count = 0
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            count += 1
    return count


def _count_species(text: str) -> int:
    matches = SPECIES_PATTERN.findall(text)
    return sum(1 for m in matches if m.split()[0] not in NON_SPECIES_WORDS)


def classify_query(query: str) -> QueryClassification:
    """Classify a query into factual/exploratory/comparative."""
    features = {
        "factual_markers": _count_matches(query, FACTUAL_MARKERS),
        "exploratory_markers": _count_matches(query, EXPLORATORY_MARKERS),
        "comparative_markers": _count_matches(query, COMPARATIVE_MARKERS),
        "species_count": _count_species(query),
        "query_length": len(query.split()),
    }

    scores = {
        "factual": features["factual_markers"] * 2.0,
        "exploratory": features["exploratory_markers"] * 1.5,
        "comparative": features["comparative_markers"] * 3.0,
    }

    if features["species_count"] >= 2:
        scores["comparative"] += 2.5

    if features["species_count"] == 1 and features["query_length"] <= 6:
        scores["factual"] += 1.5

    if features["query_length"] > 10:
        scores["exploratory"] += 1.0

    if features["exploratory_markers"] >= 2:
        scores["exploratory"] += 1.0

    total = sum(scores.values()) or 1.0
    category = max(scores, key=scores.get)
    confidence = scores[category] / total

    if confidence < 0.4:
        category = "exploratory"
        confidence = 0.5

    return QueryClassification(
        category=category,
        confidence=round(confidence, 3),
        features=features,
    )
