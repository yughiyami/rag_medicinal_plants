"""
World Flora Online GraphQL client (C4 - Capa 1: Structured data).
Validates taxonomy and normalizes scientific vs vernacular names.
~1.4M plant names under CC BY license.
"""
import time
import json
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request

from config.settings import TARGET_SPECIES, RAW_DIR

WFO_URL = "http://list.worldfloraonline.org/gql.php"

VERNACULAR_NAMES = {
    "Uncaria tomentosa": ["una de gato", "cat's claw", "garabato"],
    "Lepidium meyenii": ["maca", "maca andina", "maca peruana"],
    "Croton lechleri": ["sangre de grado", "sangre de drago", "dragon's blood"],
    "Minthostachys mollis": ["muna", "munia", "peperina"],
    "Erythroxylum coca": ["coca", "hoja de coca"],
    "Smallanthus sonchifolius": ["yacon", "llacon", "aricoma"],
    "Physalis peruviana": ["aguaymanto", "uchuva", "cape gooseberry"],
    "Buddleja incana": ["quishuar", "kiswar", "colle"],
}


def _graphql_query(query: str, variables: dict) -> dict:
    """Execute a GraphQL query against WFO."""
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    req = Request(
        WFO_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SIRCA-RAG/1.0",
        },
    )
    time.sleep(0.5)
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_name(species: str) -> list[dict]:
    """Search WFO for a species name and return matches."""
    query = """
    query SearchName($name: String!) {
        taxonNameSuggestion(terms: $name) {
            id
            fullNameString
            nameString
            authorsString
            currentPreferredUsage {
                hasName {
                    fullNameString
                }
                classification {
                    family
                    order
                    phylum
                }
            }
        }
    }
    """
    try:
        data = _graphql_query(query, {"name": species})
        suggestions = data.get("data", {}).get("taxonNameSuggestion", [])
        results = []
        for s in suggestions:
            usage = s.get("currentPreferredUsage") or {}
            preferred = usage.get("hasName", {}).get("fullNameString", "")
            classification = usage.get("classification", {})
            results.append({
                "wfo_id": s.get("id", ""),
                "full_name": s.get("fullNameString", ""),
                "name": s.get("nameString", ""),
                "authors": s.get("authorsString", ""),
                "preferred_name": preferred,
                "family": classification.get("family", ""),
                "order": classification.get("order", ""),
                "phylum": classification.get("phylum", ""),
                "source": "wfo",
            })
        return results
    except Exception as e:
        print(f"    [WFO] Error for {species}: {e}")
        return []


def acquire_wfo_all() -> dict:
    """Acquire WFO taxonomy for all target species."""
    print("\n[C4-WFO] Validating taxonomy via World Flora Online...")
    results = {"taxonomy": [], "name_mappings": []}

    for species in TARGET_SPECIES:
        print(f"  {species}...")
        matches = search_name(species)

        if matches:
            best = matches[0]
            results["taxonomy"].append(best)
            print(f"    WFO ID: {best['wfo_id']}")
            print(f"    Family: {best['family']}")
            print(f"    Preferred: {best['preferred_name']}")

            # Add vernacular name mappings
            vernaculars = VERNACULAR_NAMES.get(species, [])
            for vn in vernaculars:
                results["name_mappings"].append({
                    "scientific_name": species,
                    "vernacular_name": vn,
                    "wfo_id": best["wfo_id"],
                    "family": best["family"],
                })
        else:
            print(f"    No match found")

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"wfo_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[C4-WFO] Saved: {len(results['taxonomy'])} taxa, {len(results['name_mappings'])} name mappings -> {path}")
    return results


if __name__ == "__main__":
    acquire_wfo_all()
