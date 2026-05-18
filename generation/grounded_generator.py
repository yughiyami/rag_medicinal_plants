"""
Grounded Generation Module (C5) for SIRCA-RAG.
Generates answers strictly grounded in retrieved context with inline citations.

Supports backends:
  1. deepseek: DeepSeek V4 Flash via API (best quality, recommended for eval)
  2. ollama: Local Qwen3.5-8B via Ollama (for demo/production)
  3. local: Qwen2.5-7B-Instruct via transformers (heavy, GPU preferred)
  4. template: Template-based generation (no model needed, for testing)
"""
from dataclasses import dataclass, field

from config.settings import (
    GENERATOR_MODEL, MAX_NEW_TOKENS, TEMPERATURE,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
)


SYSTEM_PROMPT = """You are SIRCA-RAG, a scientific research assistant specialized in Peruvian medicinal plants.
You follow a strict Chain-of-Thought grounding protocol to ensure zero hallucination.

ABSOLUTE RULES:
- NEVER use prior knowledge. ONLY use what is in the context.
- NEVER paraphrase scientific terms — copy them EXACTLY from the context.
- Answer in the SAME language as the query (Spanish or English).
- If something is NOT in the context, say so. Never invent data.
"""

CONTEXT_TEMPLATE = """Context sources:
{context}

Source metadata:
{citations}

Query: {query}

Follow this Chain-of-Thought protocol strictly:

## Step 1: EXTRACT — List every relevant fact from the context
For each source [N], extract key claims as bullet points. Copy terms VERBATIM.
Include: species names, compound names, concentrations, percentages, mechanisms, assay values.

## Step 2: VERIFY — Check each extracted fact
Mark each fact with its source [N]. Discard anything not explicitly stated in context.

## Step 3: COMPOSE — Write the final answer
Using ONLY the verified facts from Steps 1-2:
- Write 3-5 sentences that directly answer the query.
- Use the EXACT words and phrases from the context — do not rephrase scientific terms.
- Cite every claim with [N].
- You MUST include ALL of these if found in context: species binomials, compound/alkaloid names, concentrations, percentages, pathway names, assay values (IC50, MIC, etc.).
- Answer in the same language as the query.

CRITICAL: Show ONLY the final answer (Step 3). Never show Steps 1-2. Do NOT start with "Based on..." or "According to..." — go straight to the scientific content."""


@dataclass
class GenerationResult:
    answer: str
    citations_used: list[int] = field(default_factory=list)
    model: str = ""
    tokens_generated: int = 0


class GroundedGenerator:
    """Generates grounded responses from retrieved context."""

    def __init__(self, backend: str = "local"):
        self._backend = backend
        self._model = None
        self._tokenizer = None

    def _load_local_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[C5] Loading {GENERATOR_MODEL} on {device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            GENERATOR_MODEL,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_MODEL,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            self._model = self._model.to(device)
        print(f"[C5] Generator ready: {GENERATOR_MODEL}")

    def generate(
        self,
        query: str,
        context: str,
        citations: list[dict],
        max_tokens: int = MAX_NEW_TOKENS,
    ) -> GenerationResult:
        """Generate a grounded answer from context."""
        citation_lines = []
        for c in citations:
            parts = [f"[{c['index']}]"]
            if c.get("title"):
                parts.append(c["title"][:80])
            if c.get("authors"):
                authors = c["authors"][:3]
                parts.append(f"({', '.join(authors)})")
            if c.get("year"):
                parts.append(str(c["year"]))
            if c.get("source"):
                parts.append(f"Source: {c['source']}")
            citation_lines.append(" | ".join(parts))

        prompt = CONTEXT_TEMPLATE.format(
            context=context,
            citations="\n".join(citation_lines),
            query=query,
        )

        if self._backend == "template":
            return self._template_generate(query, context, citations, prompt)
        elif self._backend == "deepseek":
            return self._deepseek_generate(prompt, max_tokens)
        elif self._backend == "ollama":
            return self._ollama_generate(prompt, max_tokens)

        return self._local_generate(prompt, max_tokens)

    def _local_generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        """Generate using local Qwen model."""
        self._load_local_model()
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        citations_used = _extract_citation_indices(answer)

        return GenerationResult(
            answer=answer.strip(),
            citations_used=citations_used,
            model=GENERATOR_MODEL,
            tokens_generated=len(new_tokens),
        )

    def _deepseek_generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        """Generate using DeepSeek V4 Flash API."""
        from openai import OpenAI

        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=0.9,
        )

        answer = response.choices[0].message.content.strip()
        citations_used = _extract_citation_indices(answer)
        tokens = response.usage.completion_tokens if response.usage else len(answer.split())

        return GenerationResult(
            answer=answer,
            citations_used=citations_used,
            model=f"deepseek:{DEEPSEEK_MODEL}",
            tokens_generated=tokens,
        )

    def _ollama_generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        """Generate using local Ollama (Qwen3-8B). Handles thinking mode."""
        import httpx

        think_budget = 2048
        total_predict = max_tokens + think_budget

        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}\n\n/no_think"

        r = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": full_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": total_predict,
                },
            },
            timeout=600.0,
        )
        r.raise_for_status()
        data = r.json()

        msg = data.get("message", {})
        answer = msg.get("content", "").strip()

        if not answer and msg.get("thinking"):
            answer = msg["thinking"].strip()

        citations_used = _extract_citation_indices(answer)
        tokens = data.get("eval_count", len(answer.split()))

        return GenerationResult(
            answer=answer,
            citations_used=citations_used,
            model=f"ollama:{OLLAMA_MODEL}",
            tokens_generated=tokens,
        )

    def _template_generate(
        self,
        query: str,
        context: str,
        citations: list[dict],
        prompt: str,
    ) -> GenerationResult:
        """Lightweight template-based generation for testing."""
        lines = context.split("\n\n")
        summary_parts = []
        used_citations = []

        for i, line in enumerate(lines[:5]):
            if line.startswith("["):
                idx_end = line.index("]")
                cite_idx = int(line[1:idx_end])
                content = line[idx_end + 2:].strip()
                summary_parts.append(f"{content[:200]}... [{cite_idx}]")
                used_citations.append(cite_idx)

        if not summary_parts:
            answer = "The retrieved context does not contain sufficient information to answer this query."
        else:
            answer = (
                f"Based on the available evidence:\n\n"
                + "\n\n".join(summary_parts)
            )

        return GenerationResult(
            answer=answer,
            citations_used=used_citations,
            model="template",
            tokens_generated=len(answer.split()),
        )


def _extract_citation_indices(text: str) -> list[int]:
    """Extract [N] citation references from generated text."""
    import re
    matches = re.findall(r"\[(\d+)\]", text)
    return sorted(set(int(m) for m in matches))
