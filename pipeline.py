"""
SIRCA-RAG Pipeline — Single entry point for all operations.

Usage:
  python pipeline.py acquire          # Fetch from all 7 sources (resilient, per-species)
  python pipeline.py chunk            # Deduplicate + chunk raw articles
  python pipeline.py vectorize        # Encode chunks -> FAISS + BM25 indexes
  python pipeline.py serve            # Start FastAPI web service
  python pipeline.py all              # acquire + chunk + vectorize
  python pipeline.py status           # Show corpus and vectorstore stats
"""
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    EXPANDED_SPECIES, RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR,
)


# ---------------------------------------------------------------------------
# ACQUIRE: Fetch articles from all sources with per-species error handling
# ---------------------------------------------------------------------------

def acquire(species_list: list[str] | None = None):
    species = species_list or EXPANDED_SPECIES
    print(f"[ACQUIRE] {len(species)} species, 7 sources")
    all_sources: dict[str, list] = {}

    # --- PubMed ---
    print("\n--- PubMed E-utilities ---")
    from ingestion.pubmed_client import acquire_for_species, save_raw
    seen_pmids: set[str] = set()
    pubmed_articles: list[dict] = []
    for sp in species:
        try:
            arts = acquire_for_species(sp, retmax=100)
            for a in arts:
                if a["pmid"] not in seen_pmids:
                    seen_pmids.add(a["pmid"])
                    pubmed_articles.append(a)
        except Exception as e:
            print(f"  {sp}: FAILED — {e}")
    if pubmed_articles:
        save_raw(pubmed_articles, f"pubmed_expanded_{datetime.now():%Y%m%d}.json")
    print(f"  PubMed total: {len(pubmed_articles)}")
    all_sources["pubmed"] = pubmed_articles

    # --- Semantic Scholar ---
    print("\n--- Semantic Scholar ---")
    try:
        from ingestion.semantic_scholar_client import fetch_all, save_raw as s2_save
        s2 = fetch_all(species)
        if s2:
            s2_save(s2, "expanded")
        print(f"  S2 total: {len(s2)}")
        all_sources["semantic_scholar"] = s2
    except Exception as e:
        print(f"  S2 FAILED: {e}")
        all_sources["semantic_scholar"] = []

    # --- Europe PMC ---
    print("\n--- Europe PMC ---")
    try:
        from ingestion.europe_pmc_client import fetch_all as epmc_fetch, save_raw as epmc_save
        epmc = epmc_fetch(species)
        if epmc:
            epmc_save(epmc, "expanded")
        print(f"  EPMC total: {len(epmc)}")
        all_sources["europe_pmc"] = epmc
    except Exception as e:
        print(f"  EPMC FAILED: {e}")
        all_sources["europe_pmc"] = []

    # --- CrossRef ---
    print("\n--- CrossRef ---")
    try:
        from ingestion.scielo_client import acquire_scielo_all
        cr = acquire_scielo_all(species_list=species)
        articles = cr.get("articles", [])
        print(f"  CrossRef total: {len(articles)}")
        all_sources["crossref"] = articles
    except Exception as e:
        print(f"  CrossRef FAILED: {e}")
        all_sources["crossref"] = []

    # --- GBIF ---
    print("\n--- GBIF ---")
    try:
        from ingestion.gbif_client import acquire_gbif_all
        acquire_gbif_all(species_list=species)
    except Exception as e:
        print(f"  GBIF FAILED: {e}")

    # --- PeruNPDB ---
    print("\n--- PeruNPDB ---")
    try:
        from ingestion.perunpdb_client import build_perunpdb_dataset
        build_perunpdb_dataset()
    except Exception as e:
        print(f"  PeruNPDB FAILED: {e}")

    # --- COCONUT ---
    print("\n--- COCONUT ---")
    try:
        from ingestion.coconut_client import acquire_phytochemical_all
        acquire_phytochemical_all(species_list=species)
    except Exception as e:
        print(f"  COCONUT FAILED: {e}")

    return all_sources


# ---------------------------------------------------------------------------
# CHUNK: Deduplicate articles and split into semantic chunks
# ---------------------------------------------------------------------------

def chunk(all_sources: dict[str, list] | None = None):
    if all_sources is None:
        all_sources = _load_raw_sources()

    print(f"\n[DEDUP] Deduplicating across {len(all_sources)} sources...")
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict] = []
    total = 0

    for source, articles in all_sources.items():
        if not isinstance(articles, list):
            continue
        added = 0
        for a in articles:
            total += 1
            doi = (a.get("doi") or "").strip().lower()
            title_key = (a.get("title") or "").strip().lower()[:80]
            if doi and doi in seen_dois:
                continue
            if title_key and title_key in seen_titles:
                continue
            if doi:
                seen_dois.add(doi)
            if title_key:
                seen_titles.add(title_key)
            unique.append(a)
            added += 1
        print(f"  {source}: {len(articles)} -> {added} new")

    print(f"  Total: {total} -> {len(unique)} unique articles")

    # Save combined
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = RAW_DIR / f"combined_expanded_{datetime.now():%Y%m%d}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    # Chunk
    print(f"\n[CHUNK] Splitting {len(unique)} articles...")
    from ingestion.chunker import chunk_articles
    chunks = chunk_articles(unique)
    print(f"  {len(unique)} articles -> {len(chunks)} chunks")

    out_path = PROCESSED_DIR / "chunks_expanded.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved -> {out_path}")

    return chunks


def _load_raw_sources() -> dict[str, list]:
    """Load previously acquired raw data files."""
    sources: dict[str, list] = {}
    patterns = [
        ("pubmed_expanded_*.json", "pubmed"),
        ("europe_pmc_*_expanded.json", "europe_pmc"),
        ("semantic_scholar_*_expanded.json", "semantic_scholar"),
        ("crossref_expanded_*.json", "crossref"),
    ]
    for pattern, label in patterns:
        files = sorted(RAW_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            path = files[0]
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                sources[label] = data
            elif isinstance(data, dict) and "articles" in data:
                sources[label] = data["articles"]
            print(f"  Loaded {label}: {len(sources.get(label, []))} from {path.name}")
    return sources


# ---------------------------------------------------------------------------
# VECTORIZE: Encode chunks and build FAISS + BM25 indexes
# ---------------------------------------------------------------------------

def vectorize(max_per_species: int = 100):
    import numpy as np

    chunks_path = PROCESSED_DIR / "chunks_expanded.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    print(f"[VECTORIZE] {len(all_chunks)} total chunks available")

    # Balanced subset: only species-tagged chunks
    by_species: dict[str, list] = defaultdict(list)
    for c in all_chunks:
        for sp in c.get("metadata", {}).get("species", []):
            by_species[sp].append(c)

    selected_ids: set[str] = set()
    selected: list[dict] = []
    for sp in sorted(by_species.keys()):
        count = 0
        for c in by_species[sp]:
            cid = c.get("chunk_id", str(id(c)))
            if cid not in selected_ids and count < max_per_species:
                selected_ids.add(cid)
                selected.append(c)
                count += 1

    print(f"  Balanced: {len(selected)} chunks, {len(by_species)} species, max {max_per_species}/sp")

    contents = [c["content"] for c in selected]
    metadata = [c.get("metadata", {}) for c in selected]

    # Encode with e5-multilingual (fast on CPU)
    print("  Loading multilingual-e5-base...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    prefixed = [f"passage: {t}" for t in contents]
    start = time.time()
    embeddings = model.encode(prefixed, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    elapsed = time.time() - start
    embeddings = np.array(embeddings)
    dim = 768
    del model
    gc.collect()
    print(f"  Encoded {len(embeddings)} chunks in {elapsed:.0f}s")

    # FAISS index
    import faiss
    embeddings_f32 = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings_f32)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_f32)

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

    # BM25 index
    from retrieval.bm25_index import BM25Index
    bm25 = BM25Index()
    bm25.build(contents, metadata)
    bm25.save(VECTORSTORE_DIR)

    # Info file
    info = {
        "model": "intfloat/multilingual-e5-base",
        "dimension": dim,
        "total_vectors": index.ntotal,
        "elapsed_seconds": round(elapsed, 1),
        "mode": "balanced",
        "max_per_species": max_per_species,
        "species_count": len(by_species),
        "unclassified_included": 0,
    }
    with open(VECTORSTORE_DIR / "vectorization_info.json", "w") as f:
        json.dump(info, f, indent=2)

    size_mb = (VECTORSTORE_DIR / "index.faiss").stat().st_size / (1024 * 1024)
    print(f"  FAISS: {index.ntotal} vectors, {dim}d, {size_mb:.1f}MB")
    print(f"  BM25: {index.ntotal} documents")
    return index


# ---------------------------------------------------------------------------
# STATUS: Show current corpus and vectorstore state
# ---------------------------------------------------------------------------

def status():
    print("=== SIRCA-RAG Status ===\n")

    # Raw data
    print("--- Raw Data ---")
    if RAW_DIR.exists():
        for f in sorted(RAW_DIR.glob("*.json")):
            size = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size:.1f}MB")
    else:
        print("  No raw data directory")

    # Chunks
    print("\n--- Chunks ---")
    chunks_path = PROCESSED_DIR / "chunks_expanded.json"
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        by_sp = defaultdict(int)
        for c in chunks:
            for sp in c.get("metadata", {}).get("species", []):
                by_sp[sp] += 1
        no_sp = sum(1 for c in chunks if not c.get("metadata", {}).get("species", []))
        print(f"  Total: {len(chunks)} chunks")
        print(f"  Species tagged: {len(by_sp)} species")
        print(f"  Unclassified: {no_sp}")
    else:
        print("  No chunks file")

    # Vectorstore
    print("\n--- Vectorstore ---")
    info_path = VECTORSTORE_DIR / "vectorization_info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        print(f"  Vectors: {info['total_vectors']}")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Model: {info['model']}")
        print(f"  Species: {info.get('species_count', '?')}")
        print(f"  Max/species: {info.get('max_per_species', '?')}")
    else:
        print("  No vectorstore info")

    # Species catalog
    print(f"\n--- Species Catalog ---")
    print(f"  Total: {len(EXPANDED_SPECIES)} species")


# ---------------------------------------------------------------------------
# SERVE: Start the web server
# ---------------------------------------------------------------------------

def serve():
    import uvicorn
    import os
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

COMMANDS = {
    "acquire": "Fetch articles from all 7 sources",
    "chunk": "Deduplicate + chunk raw articles",
    "vectorize": "Encode chunks -> FAISS + BM25",
    "serve": "Start FastAPI web service",
    "all": "acquire + chunk + vectorize",
    "status": "Show corpus and vectorstore stats",
}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "help" or cmd not in COMMANDS:
        print("SIRCA-RAG Pipeline\n")
        print("Commands:")
        for name, desc in COMMANDS.items():
            print(f"  {name:12s} {desc}")
        return

    start = time.time()

    if cmd == "status":
        status()
    elif cmd == "acquire":
        acquire()
    elif cmd == "chunk":
        chunk()
    elif cmd == "vectorize":
        max_per = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        vectorize(max_per_species=max_per)
    elif cmd == "serve":
        serve()
    elif cmd == "all":
        sources = acquire()
        chunk(sources)
        vectorize()

    if cmd != "serve":
        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
