"""
World Flora Online client (C4 - Capa 1: Structured data).
Validates taxonomy and normalizes scientific vs vernacular names.
~1.4M plant names under CC BY license.
Uses the WFO Plant List API v1 (REST, no auth).
"""
import ssl
import time
import json
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote_plus

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE

from config.settings import TARGET_SPECIES, EXPANDED_SPECIES, RAW_DIR

WFO_API = "https://list.worldfloraonline.org/matching_rest.php"


def search_name(species: str) -> dict | None:
    """Search WFO for a species name via the matching REST API."""
    params = {
        "input_string": species,
        "method": "full",
        "limit": 1,
    }
    url = f"{WFO_API}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
    time.sleep(0.5)

    try:
        with urlopen(req, timeout=15, context=_ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if not data or not isinstance(data, dict):
            return None

        if data.get("error"):
            return None

        match = data.get("match", {})
        if not match or not match.get("wfo_id"):
            return None

        placement = match.get("placement", "")
        parts = placement.split("/") if placement else []
        family = ""
        order = ""
        for p in parts:
            if p.endswith("aceae") or p.endswith("idae"):
                family = p
            if p.endswith("ales"):
                order = p

        parsed = data.get("parsedName", {})

        return {
            "wfo_id": match.get("wfo_id", ""),
            "full_name": match.get("full_name_plain", ""),
            "canonical": parsed.get("canonical_form", ""),
            "rank": parsed.get("rank", ""),
            "family": family,
            "order": order,
            "placement": placement,
            "source": "wfo",
        }
    except Exception as e:
        print(f"    [WFO] Error for {species}: {e}")
        return None


def acquire_wfo_all(species_list: list[str] | None = None) -> dict:
    """Acquire WFO taxonomy for all species."""
    species_list = species_list or TARGET_SPECIES
    print(f"\n[C4-WFO] Validating taxonomy via World Flora Online ({len(species_list)} species)...")
    results = {"taxonomy": []}

    for species in species_list:
        match = search_name(species)
        if match:
            results["taxonomy"].append(match)
            print(f"  {species} -> {match['wfo_id']} ({match['family']})")
        else:
            print(f"  {species} -> no match")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"wfo_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[C4-WFO] Saved: {len(results['taxonomy'])} taxa -> {path}")
    return results


if __name__ == "__main__":
    acquire_wfo_all()
