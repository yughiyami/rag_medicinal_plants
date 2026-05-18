"""
SIRCA-RAG: Full Multi-Source Acquisition Pipeline
Runs all C4 acquisition modules across 3 data layers:
  Layer 1 (Structured): PeruNPDB, COCONUT/LOTUS, GBIF, WFO
  Layer 2 (Literature): PubMed, SciELO, Unpaywall
  Layer 3 (Dynamic):    Reserved for Crawl4AI scraping (Day 5)
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import RAW_DIR, PROCESSED_DIR
from ingestion.pubmed_client import acquire_all as acquire_pubmed, save_raw
from ingestion.gbif_client import acquire_gbif_all
from ingestion.wfo_client import acquire_wfo_all
from ingestion.coconut_client import acquire_phytochemical_all
from ingestion.perunpdb_client import build_perunpdb_dataset
from ingestion.scielo_client import acquire_scielo_all
from ingestion.chunker import chunk_articles


def main():
    start = time.time()
    print("=" * 60)
    print("SIRCA-RAG: Multi-Source Acquisition Pipeline")
    print("  Peruvian Medicinal Plants Knowledge Integration")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    stats = {}

    # === LAYER 1: Structured Data ===
    print("\n" + "=" * 60)
    print("LAYER 1: STRUCTURED DATA (Knowledge Graph sources)")
    print("=" * 60)

    # PeruNPDB
    perunpdb = build_perunpdb_dataset()
    stats["perunpdb_compounds"] = len(perunpdb["compounds"])
    stats["perunpdb_triples"] = len(perunpdb["triples"])

    # GBIF
    gbif = acquire_gbif_all()
    stats["gbif_occurrences"] = len(gbif["occurrences"])
    stats["gbif_taxonomies"] = len(gbif["taxonomy"])

    # WFO
    wfo = acquire_wfo_all()
    stats["wfo_taxonomies"] = len(wfo["taxonomy"])
    stats["wfo_name_mappings"] = len(wfo["name_mappings"])

    # COCONUT + LOTUS
    phyto = acquire_phytochemical_all()
    stats["coconut_by_organism"] = len(phyto.get("coconut_by_organism", []))
    stats["coconut_by_name"] = len(phyto.get("coconut_by_name", []))
    stats["lotus_pairs"] = len(phyto.get("lotus", []))

    # === LAYER 2: Scientific Literature ===
    print("\n" + "=" * 60)
    print("LAYER 2: SCIENTIFIC LITERATURE (RAG corpus)")
    print("=" * 60)

    # PubMed (already done, but we re-run for consistency)
    pubmed_articles = acquire_pubmed(retmax_per_query=100)
    save_raw(pubmed_articles, "pubmed_full.json")
    stats["pubmed_articles"] = len(pubmed_articles)

    # SciELO
    scielo = acquire_scielo_all()
    stats["scielo_articles"] = len(scielo["articles"])
    stats["scielo_open_access"] = len(scielo["open_access"])

    # === CHUNKING: Combine all text sources ===
    print("\n" + "=" * 60)
    print("CHUNKING: Processing all text sources")
    print("=" * 60)

    # Combine PubMed + SciELO articles for chunking
    all_text_articles = pubmed_articles + scielo["articles"]

    # Also create text representations of structured data for RAG
    structured_as_text = _structured_to_text(perunpdb, gbif, wfo)
    all_text_articles.extend(structured_as_text)

    print(f"[C1] Total documents to chunk: {len(all_text_articles)}")
    chunks = chunk_articles(all_text_articles)
    stats["total_chunks"] = len(chunks)

    # Save chunks
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = PROCESSED_DIR / "chunks_full.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[C1] Saved {len(chunks)} chunks -> {chunks_path}")

    # === SUMMARY ===
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("ACQUISITION COMPLETE")
    print("=" * 60)
    print(f"\n  Layer 1 (Structured):")
    print(f"    PeruNPDB: {stats['perunpdb_compounds']} compounds, {stats['perunpdb_triples']} KG triples")
    print(f"    GBIF: {stats['gbif_occurrences']} occurrences, {stats['gbif_taxonomies']} taxonomies")
    print(f"    WFO: {stats['wfo_taxonomies']} validated taxa, {stats['wfo_name_mappings']} name mappings")
    print(f"    COCONUT: {stats['coconut_by_organism']} by organism, {stats['coconut_by_name']} by name")
    print(f"    LOTUS: {stats['lotus_pairs']} compound-organism pairs")
    print(f"\n  Layer 2 (Literature):")
    print(f"    PubMed: {stats['pubmed_articles']} articles (EN)")
    print(f"    CrossRef/SciELO: {stats['scielo_articles']} articles (bilingual)")
    print(f"    Open Access PDFs: {stats['scielo_open_access']}")
    print(f"\n  Processing:")
    print(f"    Total chunks: {stats['total_chunks']}")
    print(f"    Time elapsed: {elapsed:.1f}s")
    print("=" * 60)

    # Save stats
    stats["timestamp"] = datetime.utcnow().isoformat()
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats_path = RAW_DIR / "acquisition_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def _structured_to_text(perunpdb: dict, gbif: dict, wfo: dict) -> list[dict]:
    """
    Convert structured data into text documents for RAG indexing.
    This makes the KG data searchable via semantic retrieval.
    """
    docs = []

    # PeruNPDB compounds as text
    for comp in perunpdb["compounds"]:
        text = (
            f"{comp['compound_name']} is a {comp['compound_class']} "
            f"found in {comp['species']}. "
            f"Biological activity: {comp['biological_activity']}."
        )
        docs.append({
            "pmid": f"perunpdb_{comp['compound_name'].lower().replace(' ', '_')}",
            "doi": comp["doi"],
            "title": f"{comp['compound_name']} - {comp['species']}",
            "abstract": text,
            "authors": ["PeruNPDB"],
            "journal": "Scientific Reports",
            "year": "2023",
            "mesh_terms": [comp["compound_class"], comp["species"]],
            "content_hash": "",
            "acquired_at": "",
            "source": "perunpdb_structured",
        })

    # GBIF geographic summaries
    species_locations = {}
    for occ in gbif["occurrences"]:
        sp = occ.get("species", "")
        if sp not in species_locations:
            species_locations[sp] = {"provinces": set(), "count": 0}
        prov = occ.get("state_province", "")
        if prov:
            species_locations[sp]["provinces"].add(prov)
        species_locations[sp]["count"] += 1

    for sp, data in species_locations.items():
        provinces = ", ".join(sorted(data["provinces"])) if data["provinces"] else "various regions"
        text = (
            f"{sp} has been recorded in Peru with {data['count']} "
            f"georeferenced occurrences in: {provinces}. "
            f"Data from GBIF (Global Biodiversity Information Facility)."
        )
        docs.append({
            "pmid": f"gbif_{sp.lower().replace(' ', '_')}",
            "doi": "",
            "title": f"Geographic distribution of {sp} in Peru",
            "abstract": text,
            "authors": ["GBIF"],
            "journal": "GBIF",
            "year": "2024",
            "mesh_terms": [sp, "Peru", "biodiversity"],
            "content_hash": "",
            "acquired_at": "",
            "source": "gbif_structured",
        })

    # WFO taxonomy as text
    for tax in wfo["taxonomy"]:
        text = (
            f"{tax['full_name']} (WFO ID: {tax['wfo_id']}). "
            f"Family: {tax['family']}. Order: {tax['order']}. "
            f"Preferred name: {tax['preferred_name']}."
        )
        docs.append({
            "pmid": f"wfo_{tax['wfo_id']}",
            "doi": "",
            "title": f"Taxonomy: {tax['full_name']}",
            "abstract": text,
            "authors": ["World Flora Online"],
            "journal": "WFO",
            "year": "2024",
            "mesh_terms": [tax.get("family", ""), tax.get("order", "")],
            "content_hash": "",
            "acquired_at": "",
            "source": "wfo_structured",
        })

    print(f"  [Structured->Text] Created {len(docs)} text documents from structured data")
    return docs


if __name__ == "__main__":
    main()
