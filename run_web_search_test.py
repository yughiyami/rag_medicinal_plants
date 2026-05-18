"""
SIRCA-RAG: Day 5 — Web Search Fallback Test
Tests the Crawl4AI/httpx web searcher and its integration with the CRAG loop.
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scraping.web_searcher import WebSearcher, search_sync

print("=" * 60)
print("SIRCA-RAG: C4 Web Search Fallback Test")
print("=" * 60)

# Step 1: Test individual sources
print("\n--- Step 1: PubMed Live Search ---")
searcher = WebSearcher()

async def test_sources():
    pubmed = await searcher.search_pubmed_live(
        "Uncaria tomentosa anti-inflammatory alkaloids", max_results=3
    )
    print(f"  PubMed live: {len(pubmed)} results")
    for r in pubmed:
        print(f"    - {r.title[:70]}")
        print(f"      {r.url}")

    print("\n--- Step 2: Europe PMC Search ---")
    epmc = await searcher.search_europe_pmc(
        "Lepidium meyenii maca fertility clinical trial", max_results=3
    )
    print(f"  Europe PMC: {len(epmc)} results")
    for r in epmc:
        print(f"    - {r.title[:70]}")
        print(f"      DOI: {r.metadata.get('doi', 'N/A')}")
        print(f"      Content: {r.content[:100]}..." if r.content else "")

    print("\n--- Step 3: Combined Search ---")
    combined = await searcher.search(
        "Croton lechleri sangre de grado wound healing taspine", max_results=5
    )
    print(f"  Combined: {len(combined)} unique results")
    for r in combined:
        print(f"    [{r.source}] {r.title[:60]}")

    await searcher.close()
    return len(pubmed), len(epmc), len(combined)

counts = asyncio.run(test_sources())

# Step 4: Test URL scraping
print("\n--- Step 4: URL Scraping (httpx+BS4) ---")
searcher2 = WebSearcher()

async def test_scrape():
    result = await searcher2.scrape_url(
        "https://pubmed.ncbi.nlm.nih.gov/32145678/"
    )
    if result:
        print(f"  Title: {result.title[:80]}")
        print(f"  Content length: {len(result.content)} chars")
        print(f"  Preview: {result.content[:150]}...")
    else:
        print("  Scrape failed (expected if URL doesn't exist)")
    await searcher2.close()

asyncio.run(test_scrape())

# Step 5: Sync wrapper test
print("\n--- Step 5: Sync Wrapper ---")
sync_results = search_sync("Physalis peruviana antioxidant Peru", max_results=3)
print(f"  search_sync: {len(sync_results)} results")
for r in sync_results:
    print(f"    [{r.source}] {r.title[:60]}")

# Summary
print("\n" + "=" * 60)
pubmed_n, epmc_n, combined_n = counts
print(f"WEB SEARCH READY")
print(f"  PubMed live: {pubmed_n} | Europe PMC: {epmc_n} | Combined: {combined_n}")
print(f"  Sync wrapper: {len(sync_results)}")
print("=" * 60)
