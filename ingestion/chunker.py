"""
Semantic chunking module (C1).
Recursive character splitting that respects document structure boundaries.
Addresses the flaw in Equation 7 of the paper: uses semantic delimiters
instead of blind sliding window.
"""
import hashlib
from typing import Optional

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


def recursive_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """
    Split text recursively by semantic delimiters.
    Tries the strongest separator first (paragraph break),
    falls back to weaker ones (sentence, clause, word).
    """
    if separators is None:
        separators = CHUNK_SEPARATORS

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Find the strongest separator that exists in the text
    separator = ""
    for sep in separators:
        if sep in text:
            separator = sep
            break

    if not separator:
        # No separator found: hard split with overlap
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    # Split by the chosen separator
    parts = text.split(separator)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            # If the part itself exceeds chunk_size, recurse with weaker separators
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(separator) + 1 :]
                sub_chunks = recursive_split(part, chunk_size, overlap, remaining_seps)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    # Add overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, overlap)

    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add trailing context from previous chunk as prefix overlap."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        # Only prepend if it doesn't start mid-word
        if prev_tail and prev_tail[0] == " ":
            result.append(prev_tail.strip() + " " + chunks[i])
        else:
            result.append(chunks[i])
    return result


def chunk_article(article: dict) -> list[dict]:
    """
    Chunk a PubMed article into indexed fragments with metadata.
    Each chunk carries its provenance for citation grounding.
    """
    text = f"{article['title']}\n\n{article['abstract']}"
    if not text.strip():
        return []

    raw_chunks = recursive_split(text)

    chunks = []
    for idx, content in enumerate(raw_chunks):
        chunk_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        chunks.append({
            "chunk_id": f"{article['pmid']}_c{idx:03d}",
            "content": content,
            "content_hash": chunk_hash,
            "metadata": {
                "pmid": article["pmid"],
                "doi": article.get("doi", ""),
                "title": article["title"],
                "authors": article.get("authors", []),
                "journal": article.get("journal", ""),
                "year": article.get("year", ""),
                "species": _detect_species(text),
                "source": article.get("source", "pubmed"),
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
            },
        })

    return chunks


def _detect_species(text: str) -> list[str]:
    """Simple species detection from text."""
    from config.settings import EXPANDED_SPECIES

    text_lower = text.lower()
    found = []
    for species in EXPANDED_SPECIES:
        if species.lower() in text_lower:
            found.append(species)
    return found


def chunk_articles(articles: list[dict]) -> list[dict]:
    """Chunk a batch of articles."""
    all_chunks = []
    for article in articles:
        chunks = chunk_article(article)
        all_chunks.extend(chunks)
    return all_chunks
