"""
SciELO and open-access sources scraper (C4 extension).
Adds Spanish-language scientific literature for bilingual retrieval.
"""
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import urlencode

from config.settings import RAW_DIR, TARGET_SPECIES


SCIELO_SEARCH_URL = "https://search.scielo.org/api/v1"
WFO_GRAPHQL_URL = "http://list.worldfloraonline.org/gql.php"
GBIF_API_URL = "https://api.gbif.org/v1"


def search_scielo(query: str, limit: int = 50) -> list[dict]:
    """
    Search SciELO for Spanish/Portuguese articles.
    Uses the SciELO search API.
    """
    params = {
        "q": query,
        "lang": "es",
        "count": limit,
        "output": "json",
    }
    url = f"https://search.scielo.org/?{urlencode(params)}&format=jsonp"

    # Fallback: use OAI-PMH for reliable access
    return _search_scielo_oai(query, limit)


def _search_scielo_oai(query: str, limit: int = 50) -> list[dict]:
    """
    Access SciELO via OAI-PMH protocol.
    More reliable than the search API for programmatic access.
    """
    base_url = "https://www.scielo.br/oai/scielo-oai.php"
    params = {
        "verb": "ListRecords",
        "metadataPrefix": "oai_dc",
        "set": f"subject:{query}",
    }

    try:
        url = f"{base_url}?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
        with urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        return _parse_oai_response(content)
    except Exception as e:
        print(f"[C4-SciELO] Warning: {e}")
        return []


def _parse_oai_response(xml_text: str) -> list[dict]:
    """Parse OAI-PMH XML response."""
    import xml.etree.ElementTree as ET

    articles = []
    try:
        root = ET.fromstring(xml_text)
        ns = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "dc": "http://purl.org/dc/elements/1.1/",
        }

        for record in root.findall(".//oai:record", ns):
            metadata = record.find(".//oai:metadata", ns)
            if metadata is None:
                continue

            dc = metadata.find(".//{http://www.openarchives.org/OAI/2.0/oai_dc/}dc")
            if dc is None:
                continue

            title = dc.findtext("{http://purl.org/dc/elements/1.1/}title", "")
            description = dc.findtext("{http://purl.org/dc/elements/1.1/}description", "")
            creator = dc.findtext("{http://purl.org/dc/elements/1.1/}creator", "")
            date = dc.findtext("{http://purl.org/dc/elements/1.1/}date", "")
            identifier = dc.findtext("{http://purl.org/dc/elements/1.1/}identifier", "")

            if title and description:
                content_hash = hashlib.sha256(
                    f"{title}:{description}".encode()
                ).hexdigest()[:16]
                articles.append({
                    "pmid": f"scielo_{content_hash}",
                    "doi": identifier if "doi" in identifier.lower() else "",
                    "title": title,
                    "abstract": description,
                    "authors": [creator] if creator else [],
                    "journal": "SciELO",
                    "year": date[:4] if date else "",
                    "mesh_terms": [],
                    "content_hash": content_hash,
                    "acquired_at": datetime.utcnow().isoformat(),
                    "source": "scielo",
                    "language": "es",
                })
    except ET.ParseError:
        pass

    return articles


def fetch_gbif_occurrences(species: str, country: str = "PE", limit: int = 50) -> list[dict]:
    """
    Fetch occurrence records from GBIF for a species in Peru.
    Provides geographic distribution data for the Knowledge Graph.
    """
    params = {
        "scientificName": species,
        "country": country,
        "limit": limit,
        "hasCoordinate": "true",
    }
    url = f"{GBIF_API_URL}/occurrence/search?{urlencode(params)}"

    try:
        req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
        time.sleep(0.5)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        occurrences = []
        for result in data.get("results", []):
            occurrences.append({
                "species": result.get("species", species),
                "latitude": result.get("decimalLatitude"),
                "longitude": result.get("decimalLongitude"),
                "locality": result.get("locality", ""),
                "state_province": result.get("stateProvince", ""),
                "year": result.get("year"),
                "dataset": result.get("datasetName", ""),
                "source": "gbif",
            })
        return occurrences
    except Exception as e:
        print(f"[C4-GBIF] Warning for {species}: {e}")
        return []


def fetch_wfo_taxonomy(species: str) -> Optional[dict]:
    """
    Query World Flora Online for validated taxonomy.
    Normalizes scientific names for the knowledge graph.
    """
    query = """
    query SearchName($name: String!) {
        taxonNameSuggestion(terms: $name) {
            id
            fullNameString
            currentPreferredUsage {
                hasName { fullNameString }
                classification { family order }
            }
        }
    }
    """
    payload = json.dumps({"query": query, "variables": {"name": species}}).encode()

    try:
        req = Request(
            WFO_GRAPHQL_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SIRCA-RAG/1.0",
            },
        )
        time.sleep(0.5)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        suggestions = data.get("data", {}).get("taxonNameSuggestion", [])
        if suggestions:
            return suggestions[0]
    except Exception as e:
        print(f"[C4-WFO] Warning for {species}: {e}")

    return None


def acquire_supplementary() -> dict:
    """
    Acquire supplementary data from open sources:
    - SciELO: Spanish-language articles
    - GBIF: Geographic distribution in Peru
    - WFO: Validated taxonomy
    """
    print("\n[C4] Acquiring supplementary sources...")
    results = {"scielo": [], "gbif": [], "wfo": []}

    # SciELO articles in Spanish
    for species in TARGET_SPECIES[:4]:
        print(f"  [SciELO] {species}...")
        articles = search_scielo(f'"{species}" planta medicinal')
        results["scielo"].extend(articles)
        time.sleep(1)

    # GBIF occurrences in Peru
    for species in TARGET_SPECIES:
        print(f"  [GBIF] {species}...")
        occs = fetch_gbif_occurrences(species)
        results["gbif"].extend(occs)

    # WFO taxonomy validation
    for species in TARGET_SPECIES:
        print(f"  [WFO] {species}...")
        taxonomy = fetch_wfo_taxonomy(species)
        if taxonomy:
            results["wfo"].append({"species": species, "taxonomy": taxonomy})

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"supplementary_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[C4] Supplementary data saved → {path}")
    print(f"  SciELO: {len(results['scielo'])} articles")
    print(f"  GBIF: {len(results['gbif'])} occurrences")
    print(f"  WFO: {len(results['wfo'])} taxonomies validated")

    return results


if __name__ == "__main__":
    acquire_supplementary()
