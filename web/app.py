"""
SIRCA-RAG Web API — MVP for SimBig/WAIMLAp 2026.
FastAPI service wrapping the SIRCA-RAG pipeline.
"""
import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_agent_deepseek = None
_agent_ollama = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent_deepseek, _agent_ollama
    from agent.graph import SIRCAAgent

    from retrieval.hybrid import HybridRetriever

    print("[SIRCA-RAG] Loading retriever...")
    retriever = HybridRetriever()
    retriever.load()
    print(f"[SIRCA-RAG] Retriever ready: {retriever._faiss_index.ntotal} vectors.")

    print("[SIRCA-RAG] Loading DeepSeek agent...")
    _agent_deepseek = SIRCAAgent(retriever=retriever, generator_backend="deepseek")
    print("[SIRCA-RAG] DeepSeek agent ready.")

    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            _agent_ollama = SIRCAAgent(
                retriever=_agent_deepseek._retriever,
                generator_backend="ollama",
            )
            print("[SIRCA-RAG] Ollama agent ready (shared retriever).")
    except Exception:
        print("[SIRCA-RAG] Ollama not available, skipping.")

    yield
    print("[SIRCA-RAG] Shutting down.")


app = FastAPI(
    title="SIRCA-RAG",
    description="Semi-autonomous RAG for Peruvian Medicinal Plant Knowledge",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    backend: str = Field("deepseek", pattern="^(deepseek|ollama|template)$")
    language: str = Field("auto", pattern="^(auto|es|en)$")


class Citation(BaseModel):
    index: int
    title: str = ""
    authors: list[str] = []
    year: str = ""
    source: str = ""
    pmid: str = ""
    doi: str = ""
    species: list[str] = []


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    classification: dict
    crag_action: str
    confidence: float
    model: str
    tokens_generated: int
    latency_ms: int
    trace_summary: list[str]


class HealthResponse(BaseModel):
    status: str
    backends: dict[str, bool]
    vectorstore_size: int
    version: str


@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health", response_model=HealthResponse)
async def health():
    vs_size = 0
    try:
        if _agent_deepseek and _agent_deepseek._retriever:
            vs_size = _agent_deepseek._retriever._faiss_index.ntotal
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        backends={
            "deepseek": _agent_deepseek is not None,
            "ollama": _agent_ollama is not None,
            "template": True,
        },
        vectorstore_size=vs_size,
        version="1.0.0",
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    agent = None
    if req.backend == "deepseek":
        agent = _agent_deepseek
    elif req.backend == "ollama":
        agent = _agent_ollama
    elif req.backend == "template":
        agent = _agent_deepseek

    if agent is None:
        raise HTTPException(400, f"Backend '{req.backend}' not available")

    if req.backend == "template" and _agent_deepseek:
        from agent.graph import SIRCAAgent
        agent = SIRCAAgent(
            retriever=_agent_deepseek._retriever,
            generator_backend="template",
        )

    start = time.perf_counter()
    try:
        result = agent.run(req.query)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")
    latency = int((time.perf_counter() - start) * 1000)

    gen = result.get("generation", {})
    crag = result.get("crag_decision", {})
    classification = result.get("classification", {})

    citations = []
    for c in result.get("citations", []):
        citations.append(Citation(
            index=c.get("index", 0),
            title=c.get("title", ""),
            authors=c.get("authors", []),
            year=str(c.get("year", "")),
            source=c.get("source", ""),
            pmid=c.get("pmid", ""),
            doi=c.get("doi", ""),
            species=c.get("species", []),
        ))

    trace_summary = []
    for step in result.get("trace", []):
        node = step.get("node", "?")
        dur = step.get("duration_ms", 0)
        dur_label = "<1ms" if dur == 0 else f"{dur}ms"
        trace_summary.append(f"{node} ({dur_label})")

    return QueryResponse(
        query=result.get("query", req.query),
        answer=gen.get("answer", "No answer generated."),
        citations=citations,
        classification={
            "category": classification.get("category", "unknown"),
            "confidence": classification.get("confidence", 0),
        },
        crag_action=crag.get("action", "unknown"),
        confidence=crag.get("confidence", 0),
        model=gen.get("model", req.backend),
        tokens_generated=gen.get("tokens_generated", 0),
        latency_ms=latency,
        trace_summary=trace_summary,
    )


@app.get("/api/species")
async def list_species():
    from config.settings import TARGET_SPECIES
    return {"species": TARGET_SPECIES}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("web.app:app", host=host, port=port, reload=False)
