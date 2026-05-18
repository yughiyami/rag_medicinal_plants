"""
Run full acquisition pipeline for all target species.
Saves raw articles and processed chunks to disk.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestion.pubmed_client import acquire_all, save_raw
from ingestion.chunker import chunk_articles
from config.settings import PROCESSED_DIR


def main():
    print("=" * 60)
    print("SIRCA-RAG: Full Acquisition Run")
    print("  8 Peruvian medicinal plant species")
    print("  PubMed E-utilities API")
    print("=" * 60)

    # Acquire from PubMed
    articles = acquire_all(retmax_per_query=100)
    raw_path = save_raw(articles)

    # Chunk all articles
    print(f"\n[C1] Chunking {len(articles)} articles...")
    chunks = chunk_articles(articles)
    print(f"[C1] Generated {len(chunks)} chunks")

    # Save chunks
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = PROCESSED_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[C1] Saved -> {chunks_path}")

    # Stats
    print("\n" + "=" * 60)
    print("ACQUISITION SUMMARY")
    print(f"  Total articles: {len(articles)}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Unique sources: {len(set(a['pmid'] for a in articles))}")

    species_counts = {}
    for chunk in chunks:
        for sp in chunk["metadata"].get("species", []):
            species_counts[sp] = species_counts.get(sp, 0) + 1
    print("\n  Chunks per species:")
    for sp, count in sorted(species_counts.items(), key=lambda x: -x[1]):
        print(f"    {sp}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
