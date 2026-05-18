"""
LangGraph Agent for SIRCA-RAG (C3 Semi-Autonomous Agent).
Orchestrates the full CRAG pipeline as a state graph:

  classify -> retrieve -> evaluate -> [accept|refine|web_search] -> generate

Implements Corrective RAG with up to MAX_REFINE_ATTEMPTS query refinements
before falling back to web search or returning partial results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import StateGraph, END

from agent.query_classifier import classify_query, QueryClassification
from agent.crag_evaluator import CRAGEvaluator, CRAGDecision
from retrieval.hybrid import HybridRetriever
from generation.grounded_generator import GroundedGenerator, GenerationResult

MAX_REFINE_ATTEMPTS = 2


@dataclass
class AgentState:
    query: str = ""
    classification: QueryClassification | None = None
    retrieval_results: list[dict] = field(default_factory=list)
    crag_decision: CRAGDecision | None = None
    context: str = ""
    citations: list[dict] = field(default_factory=list)
    generation: GenerationResult | None = None
    refine_count: int = 0
    current_query: str = ""
    trace: list[dict] = field(default_factory=list)
    error: str | None = None


class SIRCAAgent:
    """LangGraph-based CRAG agent for SIRCA-RAG."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: GroundedGenerator | None = None,
        generator_backend: str = "template",
    ):
        self._retriever = retriever
        self._generator = generator or GroundedGenerator(backend=generator_backend)
        self._evaluator = CRAGEvaluator()
        self._graph = self._build_graph()

    def _ensure_retriever(self):
        if self._retriever is None:
            self._retriever = HybridRetriever()
            self._retriever.load()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        builder = StateGraph(dict)

        builder.add_node("classify", self._node_classify)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("evaluate", self._node_evaluate)
        builder.add_node("refine", self._node_refine)
        builder.add_node("web_search", self._node_web_search)
        builder.add_node("generate", self._node_generate)

        builder.set_entry_point("classify")
        builder.add_edge("classify", "retrieve")
        builder.add_edge("retrieve", "evaluate")

        builder.add_conditional_edges(
            "evaluate",
            self._route_after_eval,
            {
                "accept": "generate",
                "refine": "refine",
                "web_search": "web_search",
            },
        )

        builder.add_edge("refine", "retrieve")
        builder.add_edge("web_search", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    def _node_classify(self, state: dict) -> dict:
        """Classify the incoming query."""
        start = time.time()
        query = state["query"]
        classification = classify_query(query)

        state["classification"] = {
            "category": classification.category,
            "confidence": classification.confidence,
            "features": classification.features,
            "alpha_override": classification.alpha_override,
        }
        state["current_query"] = query
        state.setdefault("trace", []).append({
            "node": "classify",
            "category": classification.category,
            "confidence": classification.confidence,
            "duration_ms": int((time.time() - start) * 1000),
        })
        return state

    def _node_retrieve(self, state: dict) -> dict:
        """Execute hybrid retrieval."""
        start = time.time()
        self._ensure_retriever()
        query = state["current_query"]
        classification = state.get("classification", {})
        alpha_override = classification.get("alpha_override")

        if alpha_override is not None:
            original_alpha = self._retriever.alpha
            self._retriever.alpha = alpha_override

        result = self._retriever.retrieve_with_context(query, top_k=5)

        if alpha_override is not None:
            self._retriever.alpha = original_alpha

        state["retrieval_results"] = self._retriever.retrieve(query, rerank_top_k=5)
        state["context"] = result["context"]
        state["citations"] = result["citations"]
        state.setdefault("trace", []).append({
            "node": "retrieve",
            "query": query,
            "num_results": result["num_results"],
            "duration_ms": int((time.time() - start) * 1000),
        })
        return state

    def _node_evaluate(self, state: dict) -> dict:
        """CRAG evaluation of retrieval quality."""
        start = time.time()
        results = state.get("retrieval_results", [])
        query = state["current_query"]

        decision = self._evaluator.evaluate(query, results)
        state["crag_decision"] = {
            "action": decision.action,
            "confidence": decision.confidence,
            "relevant_indices": decision.relevant_indices,
            "reason": decision.reason,
            "refined_query": decision.refined_query,
        }
        state.setdefault("trace", []).append({
            "node": "evaluate",
            "action": decision.action,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "duration_ms": int((time.time() - start) * 1000),
        })
        return state

    def _route_after_eval(self, state: dict) -> Literal["accept", "refine", "web_search"]:
        """Route based on CRAG decision."""
        decision = state.get("crag_decision", {})
        action = decision.get("action", "web_search")
        refine_count = state.get("refine_count", 0)

        if action == "refine" and refine_count >= MAX_REFINE_ATTEMPTS:
            return "web_search"

        if action == "accept":
            return "accept"
        elif action == "refine":
            return "refine"
        else:
            return "web_search"

    def _node_refine(self, state: dict) -> dict:
        """Refine query and re-retrieve."""
        start = time.time()
        decision = state.get("crag_decision", {})
        refined = decision.get("refined_query", state["current_query"])
        state["current_query"] = refined
        state["refine_count"] = state.get("refine_count", 0) + 1
        state.setdefault("trace", []).append({
            "node": "refine",
            "refined_query": refined,
            "attempt": state["refine_count"],
            "duration_ms": int((time.time() - start) * 1000),
        })
        return state

    def _node_web_search(self, state: dict) -> dict:
        """
        Web search fallback — queries PubMed live + Europe PMC
        for fresh results not in our local index.
        """
        start = time.time()
        import asyncio
        from scraping.web_searcher import WebSearcher

        query = state["current_query"]
        searcher = WebSearcher()

        try:
            results = asyncio.run(searcher.search(query, max_results=5))
        except Exception as e:
            results = []
            state.setdefault("trace", []).append({
                "node": "web_search",
                "error": str(e),
                "duration_ms": int((time.time() - start) * 1000),
            })
            return state
        finally:
            asyncio.run(searcher.close())

        if results:
            web_context_parts = []
            web_citations = state.get("citations", [])
            base_idx = len(web_citations)

            for i, r in enumerate(results):
                idx = base_idx + i + 1
                content = r.content or r.snippet
                web_context_parts.append(f"[{idx}] {content}")
                web_citations.append({
                    "index": idx,
                    "title": r.title,
                    "source": f"web:{r.source}",
                    "url": r.url,
                    "pmid": r.metadata.get("pmid", ""),
                    "doi": r.metadata.get("doi", ""),
                    "year": r.metadata.get("year", ""),
                    "authors": r.metadata.get("authors", []),
                    "species": [],
                })

            existing_context = state.get("context", "")
            if existing_context:
                state["context"] = existing_context + "\n\n" + "\n\n".join(web_context_parts)
            else:
                state["context"] = "\n\n".join(web_context_parts)
            state["citations"] = web_citations

        state.setdefault("trace", []).append({
            "node": "web_search",
            "results_found": len(results),
            "sources": [r.source for r in results],
            "duration_ms": int((time.time() - start) * 1000),
        })
        return state

    def _node_generate(self, state: dict) -> dict:
        """Generate grounded response."""
        start = time.time()
        query = state["query"]
        context = state.get("context", "")
        citations = state.get("citations", [])

        if not context:
            state["generation"] = {
                "answer": "No sufficient context was found to answer this query.",
                "citations_used": [],
                "model": "none",
                "tokens_generated": 0,
            }
        else:
            result = self._generator.generate(query, context, citations)
            state["generation"] = {
                "answer": result.answer,
                "citations_used": result.citations_used,
                "model": result.model,
                "tokens_generated": result.tokens_generated,
            }

        state.setdefault("trace", []).append({
            "node": "generate",
            "model": state["generation"]["model"],
            "citations_used": state["generation"]["citations_used"],
            "duration_ms": int((time.time() - start) * 1000),
        })
        return state

    def run(self, query: str) -> dict:
        """Run the full CRAG agent pipeline."""
        initial_state = {
            "query": query,
            "classification": None,
            "retrieval_results": [],
            "crag_decision": None,
            "context": "",
            "citations": [],
            "generation": None,
            "refine_count": 0,
            "current_query": query,
            "trace": [],
            "error": None,
        }

        result = self._graph.invoke(initial_state)
        return result

    def run_batch(self, queries: list[str]) -> list[dict]:
        """Run multiple queries through the agent."""
        return [self.run(q) for q in queries]
