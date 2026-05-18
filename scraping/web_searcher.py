"""
Web Search Fallback for CRAG (C4 Extension).
Triggered when retrieval quality is insufficient.

Strategy:
  1. Google Scholar via SerpAPI-style scraping (no key needed for academic queries)
  2. PubMed live search (already have the client)
  3. Direct URL scraping via Crawl4AI or httpx+BeautifulSoup fallback

This module integrates with the agent's web_search node.
"""
import asyncio
import re
import time
from dataclasses import dataclass, field
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from config.settings import TARGET_SPECIES


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    source: str
    content: str = ""
    metadata: dict = field(default_factory=dict)


ACADEMIC_SOURCES = [
    {
        "name": "pubmed",
        "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        "params": {"db": "pubmed", "retmode": "json", "retmax": "5"},
    },
    {
        "name": "europe_pmc",
        "base": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        "params": {"format": "json", "pageSize": "5"},
    },
]

KNOWN_SCRAPE_TARGETS = [
    "https://www.sernanp.gob.pe",
    "https://www.inia.gob.pe",
    "https://scielo.org.pe",
]


class WebSearcher:
    """Lightweight web search for CRAG fallback."""

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout
        self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "User-Agent": "SIRCA-RAG/1.0 (Academic Research; mailto:gabrielbernedok75@gmail.com)"
                },
                follow_redirects=True,
            )
        return self._client

    async def search_pubmed_live(self, query: str, max_results: int = 5) -> list[WebResult]:
        """Live PubMed search for fresh results not in our index."""
        client = self._get_client()
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(max_results),
            "sort": "relevance",
        }
        try:
            r = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
            )
            r.raise_for_status()
            data = r.json()
            ids = data.get("esearchresult", {}).get("idlist", [])

            if not ids:
                return []

            fetch_r = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            )
            fetch_r.raise_for_status()
            summaries = fetch_r.json().get("result", {})

            results = []
            for pmid in ids:
                info = summaries.get(pmid, {})
                if not isinstance(info, dict):
                    continue
                results.append(WebResult(
                    title=info.get("title", ""),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    snippet=info.get("title", ""),
                    source="pubmed_live",
                    metadata={"pmid": pmid, "year": info.get("pubdate", "")[:4]},
                ))
            return results

        except Exception as e:
            print(f"[WebSearch] PubMed live error: {e}")
            return []

    async def search_europe_pmc(self, query: str, max_results: int = 5) -> list[WebResult]:
        """Europe PMC search — open access, no auth needed."""
        client = self._get_client()
        params = {
            "query": query,
            "format": "json",
            "pageSize": str(max_results),
            "resultType": "core",
        }
        try:
            r = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params=params,
            )
            r.raise_for_status()
            data = r.json()
            articles = data.get("resultList", {}).get("result", [])

            results = []
            for art in articles:
                abstract = art.get("abstractText", "")
                results.append(WebResult(
                    title=art.get("title", ""),
                    url=f"https://europepmc.org/article/{art.get('source', 'MED')}/{art.get('id', '')}",
                    snippet=abstract[:300] if abstract else art.get("title", ""),
                    source="europe_pmc",
                    content=abstract,
                    metadata={
                        "pmid": art.get("pmid", ""),
                        "doi": art.get("doi", ""),
                        "year": str(art.get("pubYear", "")),
                        "authors": [
                            a.get("fullName", "")
                            for a in art.get("authorList", {}).get("author", [])[:5]
                        ],
                    },
                ))
            return results

        except Exception as e:
            print(f"[WebSearch] Europe PMC error: {e}")
            return []

    async def scrape_url(self, url: str) -> WebResult | None:
        """Scrape a single URL using httpx + BeautifulSoup."""
        client = self._get_client()
        try:
            r = await client.get(url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            title = soup.title.string if soup.title else url
            body = soup.get_text(separator="\n", strip=True)
            body = re.sub(r"\n{3,}", "\n\n", body)
            body = body[:3000]

            return WebResult(
                title=title,
                url=url,
                snippet=body[:200],
                source="scrape",
                content=body,
            )
        except Exception as e:
            print(f"[WebSearch] Scrape error for {url}: {e}")
            return None

    async def scrape_with_crawl4ai(self, url: str) -> WebResult | None:
        """Scrape using Crawl4AI for JS-rendered pages."""
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

            config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=config)
                if result.success:
                    content = result.markdown_v2.raw_markdown[:3000] if result.markdown_v2 else result.cleaned_html[:3000]
                    return WebResult(
                        title=result.metadata.get("title", url) if result.metadata else url,
                        url=url,
                        snippet=content[:200],
                        source="crawl4ai",
                        content=content,
                    )
        except Exception as e:
            print(f"[WebSearch] Crawl4AI error for {url}: {e}")
        return None

    async def search(self, query: str, max_results: int = 5) -> list[WebResult]:
        """
        Multi-source academic web search.
        Combines PubMed live + Europe PMC for maximum coverage.
        """
        pubmed_task = self.search_pubmed_live(query, max_results=3)
        epmc_task = self.search_europe_pmc(query, max_results=3)

        pubmed_results, epmc_results = await asyncio.gather(
            pubmed_task, epmc_task, return_exceptions=True
        )

        results = []
        if isinstance(pubmed_results, list):
            results.extend(pubmed_results)
        if isinstance(epmc_results, list):
            results.extend(epmc_results)

        seen_titles = set()
        deduped = []
        for r in results:
            key = r.title.lower()[:50]
            if key not in seen_titles:
                seen_titles.add(key)
                deduped.append(r)

        return deduped[:max_results]

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


def search_sync(query: str, max_results: int = 5) -> list[WebResult]:
    """Synchronous wrapper for the web searcher."""
    async def _run():
        searcher = WebSearcher()
        try:
            return await searcher.search(query, max_results)
        finally:
            await searcher.close()

    return asyncio.run(_run())
