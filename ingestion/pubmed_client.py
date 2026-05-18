"""
PubMed E-utilities client for automated scientific acquisition (C4).
Fetches abstracts and metadata for Peruvian medicinal plant species.
"""
import time
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote_plus

from config.settings import (
    PUBMED_BASE_URL,
    PUBMED_API_KEY,
    TARGET_SPECIES,
    SEARCH_QUERIES,
    RAW_DIR,
)


def _build_url(endpoint: str, params: dict) -> str:
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    return f"{PUBMED_BASE_URL}/{endpoint}?{urlencode(params)}"


def _fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": "SIRCA-RAG/1.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _rate_limit():
    delay = 0.11 if PUBMED_API_KEY else 0.34
    time.sleep(delay)


def search_pubmed(query: str, retmax: int = 100) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    url = _build_url("esearch.fcgi", params)
    _rate_limit()
    data = json.loads(_fetch(url))
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str], batch_size: int = 50) -> list[dict]:
    """Fetch article metadata and abstracts for a list of PMIDs."""
    articles = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        url = _build_url("efetch.fcgi", params)
        _rate_limit()
        xml_text = _fetch(url)
        articles.extend(_parse_pubmed_xml(xml_text))
    return articles


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed XML response into structured records."""
    articles = []
    root = ET.fromstring(xml_text)

    for article_elem in root.findall(".//PubmedArticle"):
        medline = article_elem.find("MedlineCitation")
        if medline is None:
            continue

        pmid = medline.findtext("PMID", "")
        art = medline.find("Article")
        if art is None:
            continue

        title = art.findtext("ArticleTitle", "")

        abstract_elem = art.find("Abstract")
        abstract_parts = []
        if abstract_elem is not None:
            for text_elem in abstract_elem.findall("AbstractText"):
                label = text_elem.get("Label", "")
                text = text_elem.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        # Extract authors
        authors = []
        author_list = art.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {fore}".strip())

        # Extract journal and year
        journal_elem = art.find("Journal")
        journal = ""
        year = ""
        if journal_elem is not None:
            journal = journal_elem.findtext("Title", "")
            issue = journal_elem.find("JournalIssue")
            if issue is not None:
                pub_date = issue.find("PubDate")
                if pub_date is not None:
                    year = pub_date.findtext("Year", "")

        # Extract MeSH terms
        mesh_list = medline.find("MeshHeadingList")
        mesh_terms = []
        if mesh_list is not None:
            for heading in mesh_list.findall("MeshHeading"):
                descriptor = heading.find("DescriptorName")
                if descriptor is not None and descriptor.text:
                    mesh_terms.append(descriptor.text)

        # DOI
        doi = ""
        id_list = article_elem.find(".//ArticleIdList")
        if id_list is not None:
            for aid in id_list.findall("ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text or ""

        content_hash = hashlib.sha256(
            f"{pmid}:{title}:{abstract}".encode()
        ).hexdigest()[:16]

        articles.append({
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year,
            "mesh_terms": mesh_terms,
            "content_hash": content_hash,
            "acquired_at": datetime.utcnow().isoformat(),
            "source": "pubmed",
        })

    return articles


def acquire_for_species(
    species: str, retmax: int = 100
) -> list[dict]:
    """Run all search queries for a given species."""
    all_pmids = set()
    for query_template in SEARCH_QUERIES:
        query = query_template.format(species=species)
        pmids = search_pubmed(query, retmax=retmax)
        all_pmids.update(pmids)

    if not all_pmids:
        return []

    return fetch_abstracts(list(all_pmids))


def acquire_all(retmax_per_query: int = 100) -> list[dict]:
    """Acquire articles for all target species. Returns deduplicated list."""
    seen_pmids = set()
    all_articles = []

    for species in TARGET_SPECIES:
        print(f"[C4] Acquiring: {species}...")
        articles = acquire_for_species(species, retmax=retmax_per_query)
        for art in articles:
            if art["pmid"] not in seen_pmids:
                seen_pmids.add(art["pmid"])
                all_articles.append(art)
        print(f"  -> {len(articles)} articles (total unique: {len(all_articles)})")

    return all_articles


def save_raw(articles: list[dict], filename: Optional[str] = None) -> Path:
    """Save acquired articles to JSON."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"pubmed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path = RAW_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[C4] Saved {len(articles)} articles -> {path}")
    return path


if __name__ == "__main__":
    articles = acquire_all(retmax_per_query=50)
    save_raw(articles)
