"""Europe PMC REST API client for scientific acquisition (C4)."""
import time
import json
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote_plus

from config.settings import TARGET_SPECIES, RAW_DIR

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SEARCH_TEMPLATES = [
    '"{species}" AND (medicinal OR pharmacological OR therapeutic)',
    '"{species}" AND (bioactive OR phytochemical OR ethnobotanical)',
    '"{species}" AND Peru AND (traditional OR ethnobotany)',
]
_last_request = 0.0


def _rate_limit():
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    _last_request = time.time()


def _fetch_json(url: str) -> dict:
    _rate_limit()
    req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [EPMC] Error: {e}")
        return {}


def search_papers(query: str, page_size: int = 100, cursor: str = "*") -> tuple[list[dict], str | None]:
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "cursorMark": cursor,
        "resultType": "core",
    }
    url = f"{EPMC_BASE}/search?{urlencode(params)}"
    data = _fetch_json(url)
    results = data.get("resultList", {}).get("result", [])
    next_cursor = data.get("nextCursorMark")
    if next_cursor == cursor:
        next_cursor = None
    return results, next_cursor


def _parse_paper(paper: dict, species: str) -> dict | None:
    abstract = paper.get("abstractText", "")
    if not abstract:
        return None
    authors = []
    author_str = paper.get("authorString", "")
    if author_str:
        authors = [a.strip().rstrip(".") for a in author_str.split(",") if a.strip()]
    return {
        "title": paper.get("title", ""),
        "abstract": abstract,
        "authors": authors,
        "journal": paper.get("journalTitle", ""),
        "year": int(paper.get("pubYear", 0) or 0),
        "doi": paper.get("doi", ""),
        "pmid": paper.get("pmid", ""),
        "epmc_id": paper.get("id", ""),
        "species": [species],
        "source": "europe_pmc",
        "citation_count": int(paper.get("citedByCount", 0) or 0),
    }


def fetch_species(species: str, max_pages: int = 3) -> list[dict]:
    seen_ids = set()
    articles = []
    for template in SEARCH_TEMPLATES:
        query = template.replace("{species}", species)
        print(f"  [EPMC] Searching: {query}")
        cursor = "*"
        for page in range(max_pages):
            try:
                results, next_cursor = search_papers(query, cursor=cursor)
            except Exception as e:
                print(f"  [EPMC] Failed: {e}")
                break
            if not results:
                break
            for r in results:
                rid = r.get("id", "")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                article = _parse_paper(r, species)
                if article:
                    articles.append(article)
            if not next_cursor:
                break
            cursor = next_cursor
    return articles


def fetch_all(species_list: list[str] | None = None) -> list[dict]:
    species_list = species_list or TARGET_SPECIES
    all_articles = []
    seen_dois = set()
    for sp in species_list:
        print(f"[EPMC] Fetching: {sp}")
        articles = fetch_species(sp)
        for a in articles:
            doi = a.get("doi", "")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            all_articles.append(a)
        print(f"  [EPMC] {sp}: {len(articles)} articles")
    print(f"[EPMC] Total unique: {len(all_articles)}")
    return all_articles


def save_raw(articles: list[dict], tag: str = "") -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    fname = f"europe_pmc_{date_str}{('_' + tag) if tag else ''}.json"
    path = RAW_DIR / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[EPMC] Saved {len(articles)} articles to {path}")
    return path


if __name__ == "__main__":
    articles = fetch_all()
    save_raw(articles)
