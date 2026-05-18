"""Generate SIRCA-RAG Technical Report PDF."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

doc = SimpleDocTemplate(
    "SIRCA_RAG_Report.pdf",
    pagesize=letter,
    topMargin=0.75 * inch,
    bottomMargin=0.75 * inch,
    leftMargin=0.75 * inch,
    rightMargin=0.75 * inch,
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=22, spaceAfter=6, textColor=HexColor("#1a1d27"))
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12, textColor=HexColor("#666666"), spaceAfter=20)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceBefore=16, spaceAfter=8, textColor=HexColor("#2c3e50"))
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)
bullet = ParagraphStyle("Bullet", parent=body, leftIndent=20, bulletIndent=10)
code_style = ParagraphStyle("Code", parent=styles["Code"], fontSize=9, backColor=HexColor("#f5f5f5"), leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=4)

HEADER_BG = HexColor("#2c3e50")
GRID_COLOR = HexColor("#cccccc")
ALT_ROW = HexColor("#f9f9f9")
PASS_COLOR = HexColor("#27ae60")

TABLE_BASE = [
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, GRID_COLOR),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]

story = []

# Title
story.append(Paragraph("SIRCA-RAG", title_style))
story.append(Paragraph("Semi-Autonomous RAG for Peruvian Medicinal Plant Knowledge Integration", subtitle_style))
story.append(Paragraph("Technical Report &mdash; SimBig / WAIMLAp 2026", subtitle_style))
story.append(Spacer(1, 20))

# Section 1 — Architecture
story.append(Paragraph("1. Architecture Overview", h1))
story.append(Paragraph("The SIRCA-RAG pipeline implements a Corrective RAG (CRAG) architecture with 4 stages:", body))
story.append(Paragraph("<b>1. Query Classification</b> &mdash; Rule-based classifier (factual / exploratory / comparative) that adjusts hybrid retrieval weights dynamically.", bullet))
story.append(Paragraph("<b>2. Hybrid Retrieval</b> &mdash; BGE-M3 dense embeddings (1024d) + BM25 sparse index, fused via Reciprocal Rank Fusion (RRF), reranked with cross-encoder (ms-marco-MiniLM-L-12-v2).", bullet))
story.append(Paragraph("<b>3. CRAG Evaluation</b> &mdash; Threshold-based decision: accept (&ge; 0.35), refine (0.15&ndash;0.35), or web_search (&lt; 0.15). Uses pre-computed rerank scores for sub-millisecond evaluation.", bullet))
story.append(Paragraph("<b>4. Grounded Generation</b> &mdash; Chain-of-Thought protocol (Extract &rarr; Verify &rarr; Compose) with strict grounding rules. Supports DeepSeek V4 Flash (API), Ollama Qwen3.5 (local), and template backends.", bullet))
story.append(Spacer(1, 6))
story.append(Paragraph("Query &rarr; Classify &rarr; Retrieve &rarr; Evaluate &rarr; [accept | refine | web_search] &rarr; Generate", code_style))

# Section 2 — Data Corpus
story.append(Paragraph("2. Data Corpus", h1))
story.append(Paragraph("<b>3,040</b> vectorized chunks from <b>8 Peruvian medicinal plant species</b>.", body))
story.append(Paragraph("<b>Sources:</b> PubMed, CrossRef, PeruNPDB, GBIF, WFO, SciELO", body))
story.append(Paragraph("<b>Embedding model:</b> BAAI/bge-m3 (1024 dimensions)", body))
story.append(Paragraph("<b>Chunk size:</b> 512 tokens, overlap: 64", body))
story.append(Spacer(1, 6))

species_data = [
    ["#", "Species", "Common Name"],
    ["1", "Uncaria tomentosa", "Cat's Claw / Una de Gato"],
    ["2", "Lepidium meyenii", "Maca"],
    ["3", "Croton lechleri", "Dragon's Blood / Sangre de Grado"],
    ["4", "Minthostachys mollis", "Muna"],
    ["5", "Erythroxylum coca", "Coca"],
    ["6", "Smallanthus sonchifolius", "Yacon"],
    ["7", "Physalis peruviana", "Aguaymanto / Cape Gooseberry"],
    ["8", "Buddleja incana", "Quishuar / Kiswar"],
]
t = Table(species_data, colWidths=[30, 180, 250])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEADER_BG), ("ALIGN", (0, 0), (0, -1), "CENTER")] + TABLE_BASE))
story.append(t)

# Section 3 — Evaluation Results
story.append(Paragraph("3. Evaluation Results (DeepSeek Backend)", h1))

eval_data = [
    ["Metric", "Score", "Target", "Status"],
    ["BERTScore F1 (roberta-large)", "0.9028", "&ge; 0.90", "PASS"],
    ["Semantic Similarity (cross-encoder)", "0.9929", "&mdash;", "&mdash;"],
    ["Context Precision", "1.0000", "&mdash;", "&mdash;"],
    ["Context Recall", "1.0000", "&mdash;", "&mdash;"],
    ["MRR", "1.0000", "&mdash;", "&mdash;"],
    ["NDCG@5", "1.0000", "&mdash;", "&mdash;"],
    ["Entity Recall", "0.7946", "&mdash;", "&mdash;"],
    ["Faithfulness", "0.8411", "&ge; 0.80", "PASS"],
    ["Answer Relevancy", "0.9995", "&mdash;", "&mdash;"],
]
t2 = Table(eval_data, colWidths=[220, 80, 80, 60])
t2.setStyle(TableStyle(
    [("BACKGROUND", (0, 0), (-1, 0), HEADER_BG), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
     ("TEXTCOLOR", (3, 1), (3, 1), PASS_COLOR), ("FONTNAME", (3, 1), (3, 1), "Helvetica-Bold"),
     ("TEXTCOLOR", (3, 8), (3, 8), PASS_COLOR), ("FONTNAME", (3, 8), (3, 8), "Helvetica-Bold")]
    + TABLE_BASE
))
story.append(t2)

# Section 4 — Technical Decisions
story.append(PageBreak())
story.append(Paragraph("4. Key Technical Decisions", h1))
story.append(Paragraph("<b>Chain-of-Thought Grounding:</b> 3-step protocol prevents data invention. The model internally (1) extracts facts from context, (2) verifies each against sources, (3) composes the final answer. Only Step 3 is shown to the user.", body))
story.append(Paragraph("<b>Hybrid Faithfulness Metric:</b> 65% semantic (cross-encoder sentence-level scoring via ms-marco-MiniLM-L-12-v2 with sigmoid normalization) + 35% lexical (content-word overlap with bilingual stopword filtering EN+ES).", body))
story.append(Paragraph("<b>BERTScore with roberta-large:</b> DeBERTa variants caused OverflowError on this system. roberta-large provides stable F1 &ge; 0.90.", body))
story.append(Paragraph("<b>Temperature 0.0:</b> Deterministic generation for reproducible evaluation across runs.", body))
story.append(Paragraph("<b>Bilingual Support:</b> System prompt instructs model to answer in the same language as the query (Spanish or English). Boilerplate filtering handles both languages.", body))

# Section 5 — Pipeline Trace
story.append(Paragraph("5. Pipeline Trace Example", h1))
story.append(Paragraph('Query: "What are the main alkaloids in Uncaria tomentosa?"', body))

trace_data = [
    ["Node", "Duration", "Details"],
    ["classify", "4ms", "category: exploratory, confidence: 0.60"],
    ["retrieve", "3,195ms", "5 documents, hybrid alpha: 0.6, BGE-M3 + BM25 + rerank"],
    ["evaluate", "<1ms", "action: accept, confidence: 0.89 (pre-computed scores)"],
    ["generate", "varies", "DeepSeek: ~30-150s | Template: <1ms | Ollama: ~10-30s"],
]
t3 = Table(trace_data, colWidths=[70, 80, 310])
t3.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEADER_BG)] + TABLE_BASE))
story.append(t3)

# Section 6 — Web Service
story.append(Paragraph("6. Web Service API", h1))
api_data = [
    ["Method", "Path", "Description"],
    ["GET", "/", "Interactive dark-themed frontend with example queries"],
    ["GET", "/api/health", "System status, backend availability, vectorstore size"],
    ["POST", "/api/query", "Full pipeline: query, backend, language &rarr; answer, citations, trace"],
    ["GET", "/api/species", "List of 8 target species"],
]
t4 = Table(api_data, colWidths=[60, 100, 300])
t4.setStyle(TableStyle(
    [("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
     ("FONTNAME", (0, 1), (0, -1), "Courier"),
     ("FONTNAME", (1, 1), (1, -1), "Courier")]
    + TABLE_BASE
))
story.append(t4)

# Section 7 — Deployment
story.append(Paragraph("7. Deployment (Docker / Dokploy)", h1))
story.append(Paragraph("<b>Image:</b> python:3.11-slim", body))
story.append(Paragraph("<b>Port:</b> 8000", body))
story.append(Paragraph("<b>Environment variables:</b> DEEPSEEK_API_KEY, OLLAMA_BASE_URL", body))
story.append(Paragraph("<b>Health check:</b> GET /api/health (30s interval, 60s start period)", body))
story.append(Spacer(1, 8))
story.append(Paragraph("docker compose up --build", code_style))

doc.build(story)
print("PDF created: SIRCA_RAG_Report.pdf")
