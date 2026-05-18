"""
GBIF API client (C4 - Capa 1: Structured data).
Fetches georeferenced occurrence records for Peruvian medicinal plants.
~3.1B occurrences, CC0/CC-BY license.
"""
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import urlencode

from config.settings import TARGET_SPECIES, RAW_DIR

GBIF_API = "https://api.gbif.org/v1"


def search_species_key(species: str) -> Optional[int]:
    """Resolve a species name to a GBIF taxon key."""
    params = {"name": species, "limit": 5}
    url = f"{GBIF_API}/species/match?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
    time.sleep(0.3)
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("matchType") != "NONE" and data.get("usageKey"):
        return data["usageKey"]
    return None


def fetch_occurrences(
    species: str,
    country: str = "PE",
    limit: int = 300,
) -> list[dict]:
    """Fetch occurrence records from GBIF for a species in Peru."""
    records = []
    offset = 0
    batch = min(limit, 300)

    while offset < limit:
        params = {
            "scientificName": species,
            "country": country,
            "limit": batch,
            "offset": offset,
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
        }
        url = f"{GBIF_API}/occurrence/search?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
        time.sleep(0.4)

        try:
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"    [GBIF] Error: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            records.append({
                "species": r.get("species", species),
                "scientific_name": r.get("scientificName", ""),
                "latitude": r.get("decimalLatitude"),
                "longitude": r.get("decimalLongitude"),
                "locality": r.get("locality", ""),
                "state_province": r.get("stateProvince", ""),
                "country": r.get("country", ""),
                "year": r.get("year"),
                "basis_of_record": r.get("basisOfRecord", ""),
                "dataset_name": r.get("datasetName", ""),
                "institution": r.get("institutionCode", ""),
                "gbif_id": r.get("key"),
                "source": "gbif",
            })

        offset += len(results)
        if data.get("endOfRecords", True):
            break

    return records


def fetch_species_details(taxon_key: int) -> Optional[dict]:
    """Fetch detailed taxonomic info for a species."""
    url = f"{GBIF_API}/species/{taxon_key}"
    req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
    time.sleep(0.3)
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            "taxon_key": taxon_key,
            "scientific_name": data.get("scientificName", ""),
            "canonical_name": data.get("canonicalName", ""),
            "kingdom": data.get("kingdom", ""),
            "phylum": data.get("phylum", ""),
            "class": data.get("class", ""),
            "order": data.get("order", ""),
            "family": data.get("family", ""),
            "genus": data.get("genus", ""),
            "species": data.get("species", ""),
            "taxonomic_status": data.get("taxonomicStatus", ""),
            "source": "gbif",
        }
    except Exception as e:
        print(f"    [GBIF] Species detail error: {e}")
        return None


def acquire_gbif_all() -> dict:
    """Acquire GBIF data for all target species."""
    print("\n[C4-GBIF] Acquiring geographic occurrences in Peru...")
    results = {"occurrences": [], "taxonomy": []}

    for species in TARGET_SPECIES:
        print(f"  {species}...")

        # Taxonomy
        key = search_species_key(species)
        if key:
            details = fetch_species_details(key)
            if details:
                results["taxonomy"].append(details)
                print(f"    Taxonomy: {details.get('family', '?')} / {details.get('order', '?')}")

        # Occurrences in Peru
        occs = fetch_occurrences(species, country="PE", limit=200)
        results["occurrences"].extend(occs)
        print(f"    Occurrences PE: {len(occs)}")

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"gbif_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_occ = len(results["occurrences"])
    total_tax = len(results["taxonomy"])
    print(f"\n[C4-GBIF] Saved: {total_occ} occurrences, {total_tax} taxonomies -> {path}")
    return results


if __name__ == "__main__":
    acquire_gbif_all()
