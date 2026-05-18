"""
COCONUT 2.0 + LOTUS client (C4 - Capa 1: Structured data).
Natural products databases for phytochemical entities.
COCONUT: POST /api/search (public, no auth needed).
LOTUS: Wikidata SPARQL endpoint for compound-organism pairs.
"""
import time
import json
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import urlencode

from config.settings import TARGET_SPECIES, RAW_DIR

COCONUT_SEARCH = "https://coconut.naturalproducts.net/api/search"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def _coconut_post(payload: dict) -> dict | None:
    """POST to COCONUT /api/search. Response is paginated: data.data has items."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(COCONUT_SEARCH, data=body, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "SIRCA-RAG/1.0",
    })
    time.sleep(0.8)
    try:
        with urlopen(req, timeout=20) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        inner = raw.get("data", raw)
        if isinstance(inner, dict) and "data" in inner:
            return {"items": inner["data"], "total": inner.get("total", 0)}
        if isinstance(inner, list):
            return {"items": inner, "total": len(inner)}
        return {"items": [], "total": 0}
    except HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")[:200]
        print(f"    [COCONUT] {e.code}: {body_text}")
        return None
    except Exception as e:
        print(f"    [COCONUT] {e}")
        return None


def search_coconut_by_organism(species: str, limit: int = 24) -> list[dict]:
    """Tag-based search by organism name."""
    result = _coconut_post({
        "type": "tags",
        "tagType": "organisms",
        "query": species,
        "limit": limit,
        "page": 1,
        "offset": 0,
    })
    if not result:
        return []
    return [_normalize(item, species) for item in result["items"]]


def search_coconut_basic(query: str, limit: int = 24) -> list[dict]:
    """Basic name/SMILES/InChI search (no type field)."""
    result = _coconut_post({"query": query, "limit": limit, "page": 1, "offset": 0})
    if not result:
        return []
    return [_normalize(item, query) for item in result["items"]]


def search_coconut_by_collection(collection: str, limit: int = 24) -> list[dict]:
    """Tag-based search by data source/collection name."""
    result = _coconut_post({
        "type": "tags",
        "tagType": "dataSource",
        "query": collection,
        "limit": limit,
        "page": 1,
        "offset": 0,
    })
    if not result:
        return []
    return [_normalize(item, collection) for item in result["items"]]


def search_coconut_alkaloids(limit: int = 24) -> list[dict]:
    """Filter search for alkaloids (common in Peruvian medicinal plants)."""
    result = _coconut_post({
        "type": "filters",
        "tagType": "",
        "query": "np_superclass:Alkaloids",
        "limit": limit,
        "page": 1,
        "offset": 0,
    })
    if not result:
        return []
    return [_normalize(item, "alkaloids") for item in result["items"]]


def _normalize(item: dict, query_context: str) -> dict:
    return {
        "coconut_id": item.get("identifier", ""),
        "name": item.get("name", ""),
        "iupac_name": item.get("iupac_name", ""),
        "canonical_smiles": item.get("canonical_smiles", ""),
        "organism_count": item.get("organism_count", 0),
        "citation_count": item.get("citation_count", 0),
        "collection_count": item.get("collection_count", 0),
        "annotation_level": item.get("annotation_level", 0),
        "query_context": query_context,
        "source": "coconut",
    }


def query_lotus_wikidata(species: str, limit: int = 50) -> list[dict]:
    """Query LOTUS via Wikidata SPARQL for compound-organism pairs."""
    sparql = f"""
    SELECT ?compound ?compoundLabel ?smiles ?taxon ?taxonLabel WHERE {{
      ?taxon wdt:P225 "{species}" .
      ?compound wdt:P703 ?taxon .
      OPTIONAL {{ ?compound wdt:P233 ?smiles . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,es" . }}
    }}
    LIMIT {limit}
    """

    params = {"query": sparql, "format": "json"}
    url = f"{WIKIDATA_SPARQL}?{urlencode(params)}"
    req = Request(url, headers={
        "User-Agent": "SIRCA-RAG/1.0 (mailto:dmarron@unsa.edu.pe)",
        "Accept": "application/json",
    })
    time.sleep(2.0)

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        for b in data.get("results", {}).get("bindings", []):
            results.append({
                "compound_uri": b.get("compound", {}).get("value", ""),
                "compound_name": b.get("compoundLabel", {}).get("value", ""),
                "smiles": b.get("smiles", {}).get("value", ""),
                "taxon_name": b.get("taxonLabel", {}).get("value", species),
                "source": "lotus_wikidata",
            })
        return results
    except Exception as e:
        print(f"    [LOTUS] {e}")
        return []


# Known compound names from PeruNPDB literature for direct COCONUT lookup
KNOWN_COMPOUNDS = [
    "mitraphylline", "isomitraphylline", "pteropodine", "rhynchophylline",
    "taspine", "macamides", "withanolide", "pulegone", "thymol",
    "chlorogenic acid", "catechin",
]


def acquire_phytochemical_all() -> dict:
    """Acquire phytochemical data from COCONUT and LOTUS for all species."""
    print("\n[C4-PHYTO] Acquiring phytochemical data...")
    results = {"coconut_by_organism": [], "coconut_by_name": [], "lotus": []}

    # Strategy 1: Search COCONUT by organism name
    for species in TARGET_SPECIES:
        print(f"  {species}...")
        mols = search_coconut_by_organism(species, limit=20)
        results["coconut_by_organism"].extend(mols)
        print(f"    COCONUT by organism: {len(mols)}")

        # LOTUS
        lotus = query_lotus_wikidata(species, limit=30)
        results["lotus"].extend(lotus)
        print(f"    LOTUS: {len(lotus)} pairs")

    # Strategy 2: Search COCONUT by known compound names
    print("\n  Searching known compounds by name...")
    for compound in KNOWN_COMPOUNDS:
        mols = search_coconut_basic(compound, limit=5)
        results["coconut_by_name"].extend(mols)
        if mols:
            print(f"    {compound}: {len(mols)} hits")

    # Strategy 3: Try PeruNPDB collection
    print("\n  Searching PeruNPDB collection...")
    perunpdb_mols = search_coconut_by_collection("PeruNPDB", limit=24)
    if perunpdb_mols:
        results["coconut_by_organism"].extend(perunpdb_mols)
        print(f"    PeruNPDB collection: {len(perunpdb_mols)} molecules")

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"phytochemical_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_org = len(results["coconut_by_organism"])
    n_name = len(results["coconut_by_name"])
    n_lotus = len(results["lotus"])
    print(f"\n[C4-PHYTO] Saved: {n_org} by organism, {n_name} by name, {n_lotus} LOTUS -> {path}")
    return results


if __name__ == "__main__":
    acquire_phytochemical_all()
