"""
SciELO + CrossRef client (C4 - Capa 2: Bilingual literature).
Strategy:
  1. CrossRef API - reliable, has abstracts, multilingual
  2. SciELO ArticleMeta API - Peru collection articles
  3. Unpaywall - OA PDF availability
"""
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import urlencode

from config.settings import TARGET_SPECIES, RAW_DIR

CROSSREF_API = "https://api.crossref.org/works"
ARTICLEMETA_API = "https://articlemeta.scielo.org/api/v1"
CONTACT_EMAIL = "dmarron@unsa.edu.pe"


def search_crossref(query: str, limit: int = 20, filters: str = "has-abstract:true") -> list[dict]:
    """
    Search CrossRef for articles. Reliable, no auth needed.
    Polite pool: include mailto for faster rate limits.
    """
    params = {
        "query": query,
        "rows": limit,
        "filter": filters,
        "mailto": CONTACT_EMAIL,
    }
    url = f"{CROSSREF_API}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": f"SIRCA-RAG/1.0 (mailto:{CONTACT_EMAIL})"})
    time.sleep(0.5)

    try:
        with urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("message", {}).get("items", [])
        return [_normalize_crossref(item) for item in items]
    except Exception as e:
        print(f"    [CrossRef] {e}")
        return []


def _normalize_crossref(item: dict) -> dict:
    title = item.get("title", [""])[0] if item.get("title") else ""
    abstract = item.get("abstract", "")
    # CrossRef abstracts sometimes have JATS XML tags
    abstract = abstract.replace("<jats:p>", "").replace("</jats:p>", "")
    abstract = abstract.replace("<jats:italic>", "").replace("</jats:italic>", "")

    authors = []
    for auth in item.get("author", []):
        name = f"{auth.get('family', '')} {auth.get('given', '')}".strip()
        if name:
            authors.append(name)

    doi = item.get("DOI", "")
    journal = item.get("container-title", [""])[0] if item.get("container-title") else ""

    # Extract year from published-print or published-online
    year = ""
    for date_field in ["published-print", "published-online"]:
        date_parts = item.get(date_field, {}).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = str(date_parts[0][0])
            break

    lang = item.get("language", "")
    subjects = item.get("subject", [])

    content_hash = hashlib.sha256(f"{doi}:{title}".encode()).hexdigest()[:16]

    return {
        "pmid": f"crossref_{content_hash}",
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "year": year,
        "mesh_terms": subjects,
        "content_hash": content_hash,
        "acquired_at": datetime.utcnow().isoformat(),
        "source": "crossref",
        "language": lang,
    }


def search_scielo_articlemeta(collection: str = "per", limit: int = 50) -> list[str]:
    """Get article PIDs from SciELO ArticleMeta for a collection."""
    url = f"{ARTICLEMETA_API}/article/identifiers/?collection={collection}&limit={limit}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "SIRCA-RAG/1.0"})
    time.sleep(0.5)

    try:
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [obj["code"] for obj in data.get("objects", []) if obj.get("code")]
    except Exception as e:
        print(f"    [ArticleMeta] {e}")
        return []


def fetch_articlemeta_detail(pid: str, collection: str = "per") -> dict | None:
    """Fetch full article from ArticleMeta by PID."""
    url = f"{ARTICLEMETA_API}/article/?code={pid}&collection={collection}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "SIRCA-RAG/1.0"})
    time.sleep(0.3)

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        title, abstract = "", ""
        for lang in ["es", "en", "pt"]:
            if not title and lang in data.get("title", {}):
                title = data["title"][lang]
            if not abstract and lang in data.get("abstract", {}):
                abstract = data["abstract"][lang]

        if not title:
            return None

        authors = []
        for auth in data.get("authors", []):
            name = f"{auth.get('surname', '')} {auth.get('given_names', '')}".strip()
            if name:
                authors.append(name)

        doi = data.get("doi", "")
        year = data.get("publication_date", "")[:4]
        journal = data.get("journal", {}).get("title", "")
        lang = data.get("original_language", "es")
        keywords = data.get("keywords", {}).get("es", []) or data.get("keywords", {}).get("en", [])

        content_hash = hashlib.sha256(f"{pid}:{title}".encode()).hexdigest()[:16]

        return {
            "pmid": f"scielo_{pid}",
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
            "mesh_terms": keywords,
            "content_hash": content_hash,
            "acquired_at": datetime.utcnow().isoformat(),
            "source": "scielo_articlemeta",
            "language": lang,
        }
    except Exception:
        return None


def search_unpaywall(doi: str) -> dict | None:
    """Check Unpaywall for open access PDF availability."""
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}?email={CONTACT_EMAIL}"
    req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
    time.sleep(0.3)
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("is_oa"):
            best = data.get("best_oa_location", {})
            return {
                "doi": doi,
                "is_oa": True,
                "oa_url": best.get("url", ""),
                "pdf_url": best.get("url_for_pdf", ""),
                "license": best.get("license", ""),
                "source": "unpaywall",
            }
    except Exception:
        pass
    return None


def acquire_scielo_all() -> dict:
    """Acquire bilingual literature via CrossRef + SciELO ArticleMeta."""
    print("\n[C4-Literature] Acquiring bilingual scientific literature...")
    results = {"articles": [], "open_access": []}
    seen_hashes = set()

    # Strategy 1: CrossRef (most reliable, multilingual)
    queries = [
        '"{species}" medicinal Peru',
        '"{species}" pharmacological ethnobotanical',
        '"{species}" fitoquimica planta medicinal',
    ]

    for species in TARGET_SPECIES:
        print(f"  {species}...")
        species_count = 0

        for q_template in queries:
            query = q_template.format(species=species)
            articles = search_crossref(query, limit=15)
            for art in articles:
                if art["content_hash"] not in seen_hashes:
                    seen_hashes.add(art["content_hash"])
                    results["articles"].append(art)
                    species_count += 1

        print(f"    CrossRef: {species_count} unique articles")

    # Strategy 2: SciELO ArticleMeta (Peru collection sample)
    print("\n  [SciELO ArticleMeta] Fetching Peru collection sample...")
    pids = search_scielo_articlemeta("per", limit=30)
    scielo_count = 0
    for pid in pids[:30]:
        art = fetch_articlemeta_detail(pid, "per")
        if art and art["content_hash"] not in seen_hashes:
            # Check if it's related to medicinal plants
            text = f"{art['title']} {art['abstract']}".lower()
            plant_keywords = ["planta", "medicinal", "farmaco", "ethnobota", "fitoquim",
                            "herbal", "traditional", "natural product"]
            if any(kw in text for kw in plant_keywords):
                seen_hashes.add(art["content_hash"])
                results["articles"].append(art)
                scielo_count += 1

    print(f"    SciELO ArticleMeta: {scielo_count} relevant articles from Peru")

    # Strategy 3: Unpaywall for OA PDFs
    dois = [a["doi"] for a in results["articles"] if a.get("doi")]
    print(f"\n  [Unpaywall] Checking {len(dois)} DOIs for OA...")
    for doi in dois[:30]:
        oa = search_unpaywall(doi)
        if oa:
            results["open_access"].append(oa)
    print(f"    Open Access PDFs: {len(results['open_access'])}")

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"literature_bilingual_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(results["articles"])
    langs = {}
    for a in results["articles"]:
        l = a.get("language", "?")
        langs[l] = langs.get(l, 0) + 1

    print(f"\n[C4-Literature] Total: {total} articles")
    print(f"  Languages: {langs}")
    print(f"  Saved -> {path}")
    return results


if __name__ == "__main__":
    acquire_scielo_all()
