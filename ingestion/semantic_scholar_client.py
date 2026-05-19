"""Semantic Scholar API client for scientific acquisition (C4)."""
import time
import json
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.parse import urlencode

from config.settings import TARGET_SPECIES, RAW_DIR

S2_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,authors,year,journal,externalIds,citationCount"
SEARCH_TEMPLATES = [
    '"{species}" medicinal pharmacological',
    '"{species}" bioactive phytochemical',
    '"{species}" ethnobotanical therapeutic Peru',
]
_last_request = 0.0


def _rate_limit():
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < 3.2:
        time.sleep(3.2 - elapsed)
    _last_request = time.time()


def _fetch_json(url: str) -> dict:
    _rate_limit()
    req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [S2] Error: {e}")
        return {}


def search_papers(query: str, limit: int = 100, offset: int = 0) -> list[dict]:
    params = {
        "query": query,
        "limit": min(limit, 100),
        "offset": offset,
        "fields": FIELDS,
    }
    url = f"{S2_BASE}/paper/search?{urlencode(params)}"
    data = _fetch_json(url)
    return data.get("data", [])


def _parse_paper(paper: dict, species: str) -> dict | None:
    if not paper.get("abstract"):
        return None
    ext_ids = paper.get("externalIds") or {}
    authors = []
    for a in (paper.get("authors") or []):
        if a.get("name"):
            authors.append(a["name"])
    journal = ""
    if paper.get("journal"):
        journal = paper["journal"].get("name", "")
    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "authors": authors,
        "journal": journal,
        "year": paper.get("year") or 0,
        "doi": ext_ids.get("DOI", ""),
        "pmid": ext_ids.get("PubMed", ""),
        "s2id": paper.get("paperId", ""),
        "species": [species],
        "source": "semantic_scholar",
        "citation_count": paper.get("citationCount", 0),
    }


def fetch_species(species: str) -> list[dict]:
    seen_ids = set()
    articles = []
    for template in SEARCH_TEMPLATES:
        query = template.replace("{species}", species)
        print(f"  [S2] Searching: {query}")
        try:
            papers = search_papers(query, limit=100)
        except Exception as e:
            print(f"  [S2] Failed: {e}")
            continue
        for p in papers:
            pid = p.get("paperId", "")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            article = _parse_paper(p, species)
            if article:
                articles.append(article)
    return articles


def fetch_all(species_list: list[str] | None = None) -> list[dict]:
    species_list = species_list or TARGET_SPECIES
    all_articles = []
    seen_dois = set()
    for sp in species_list:
        print(f"[S2] Fetching: {sp}")
        articles = fetch_species(sp)
        for a in articles:
            doi = a.get("doi", "")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            all_articles.append(a)
        print(f"  [S2] {sp}: {len(articles)} articles")
    print(f"[S2] Total unique: {len(all_articles)}")
    return all_articles


def save_raw(articles: list[dict], tag: str = "") -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    fname = f"semantic_scholar_{date_str}{('_' + tag) if tag else ''}.json"
    path = RAW_DIR / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[S2] Saved {len(articles)} articles to {path}")
    return path


if __name__ == "__main__":
    articles = fetch_all()
    save_raw(articles)
