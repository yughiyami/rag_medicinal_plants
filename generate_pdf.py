"""Generate SIRCA-RAG Technical Report PDF with diagrams and charts."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Flowable,
)
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF

WIDTH, HEIGHT = letter

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
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6, textColor=HexColor("#34495e"))
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)
bullet = ParagraphStyle("Bullet", parent=body, leftIndent=20, bulletIndent=10)
code_style = ParagraphStyle("Code", parent=styles["Code"], fontSize=9, backColor=HexColor("#f5f5f5"), leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=4)
caption_style = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9, textColor=HexColor("#888888"), alignment=1, spaceAfter=12, spaceBefore=4)

HEADER_BG = HexColor("#2c3e50")
GRID_COLOR = HexColor("#cccccc")
ALT_ROW = HexColor("#f9f9f9")
PASS_COLOR = HexColor("#27ae60")

PALETTE = {
    "blue": HexColor("#3498db"),
    "purple": HexColor("#9b59b6"),
    "orange": HexColor("#e67e22"),
    "red": HexColor("#e74c3c"),
    "green": HexColor("#27ae60"),
    "green_light": HexColor("#2ecc71"),
    "yellow": HexColor("#f1c40f"),
    "gray": HexColor("#95a5a6"),
    "dark": HexColor("#2c3e50"),
    "white": HexColor("#ffffff"),
}

TABLE_BASE = [
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, GRID_COLOR),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]


def _rounded_box(d, x, y, w, h, fill, label, font_size=9):
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=None, rx=6, ry=6))
    d.add(String(x + w / 2, y + h / 2 - font_size / 3, label,
                 fontSize=font_size, fillColor=PALETTE["white"],
                 textAnchor="middle", fontName="Helvetica-Bold"))


def _arrow(d, x1, y1, x2, y2, color=PALETTE["dark"]):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.5))
    size = 5
    if x2 > x1:
        d.add(Polygon(points=[x2, y2, x2 - size, y2 + size / 2, x2 - size, y2 - size / 2],
                       fillColor=color, strokeColor=None))
    elif y2 < y1:
        d.add(Polygon(points=[x2, y2, x2 - size / 2, y2 + size, x2 + size / 2, y2 + size],
                       fillColor=color, strokeColor=None))
    elif y2 > y1:
        d.add(Polygon(points=[x2, y2, x2 - size / 2, y2 - size, x2 + size / 2, y2 - size],
                       fillColor=color, strokeColor=None))


def build_architecture_diagram():
    d = Drawing(480, 200)
    bw, bh = 90, 32
    y_main = 110
    nodes = [
        (10, y_main, bw, bh, PALETTE["blue"], "Query"),
        (120, y_main, bw, bh, PALETTE["purple"], "Classify"),
        (230, y_main, bw, bh, PALETTE["orange"], "Retrieve"),
        (340, y_main, bw, bh, PALETTE["red"], "Evaluate"),
    ]
    for x, y, w, h, c, label in nodes:
        _rounded_box(d, x, y, w, h, c, label)

    _arrow(d, 100, y_main + bh / 2, 120, y_main + bh / 2)
    _arrow(d, 210, y_main + bh / 2, 230, y_main + bh / 2)
    _arrow(d, 320, y_main + bh / 2, 340, y_main + bh / 2)

    _rounded_box(d, 230, 30, bw, bh, PALETTE["green"], "Generate")

    decisions = [
        (350, 80, "accept"),
        (350, 60, "refine"),
        (350, 40, "web_search"),
    ]
    for x, y, label in decisions:
        d.add(String(x, y, label, fontSize=8, fillColor=PALETTE["dark"], fontName="Helvetica"))

    _arrow(d, 385, y_main, 385, 72)
    _arrow(d, 340, 46, 320, 46)

    _rounded_box(d, 100, 30, bw, bh, PALETTE["green_light"], "Answer")
    _arrow(d, 230, 46, 190, 46)

    d.add(String(240, 185, "SIRCA-RAG Pipeline Architecture", fontSize=11,
                 fillColor=PALETTE["dark"], textAnchor="middle", fontName="Helvetica-Bold"))
    return d


def build_crag_decision_diagram():
    d = Drawing(480, 170)
    _rounded_box(d, 180, 120, 120, 30, PALETTE["orange"], "Rerank Score")

    y_dec = 60
    _rounded_box(d, 20, y_dec, 100, 28, PALETTE["green"], "ACCEPT")
    _rounded_box(d, 180, y_dec, 100, 28, PALETTE["yellow"], "REFINE")
    _rounded_box(d, 350, y_dec, 100, 28, PALETTE["red"], "WEB SEARCH")

    d.add(String(70, y_dec + 35, ">= 0.35", fontSize=8, fillColor=PALETTE["dark"], textAnchor="middle"))
    d.add(String(230, y_dec + 35, "0.15 - 0.35", fontSize=8, fillColor=PALETTE["dark"], textAnchor="middle"))
    d.add(String(400, y_dec + 35, "< 0.15", fontSize=8, fillColor=PALETTE["dark"], textAnchor="middle"))

    _arrow(d, 210, 120, 70, y_dec + 28 + 5)
    _arrow(d, 240, 120, 230, y_dec + 28 + 5)
    _arrow(d, 270, 120, 400, y_dec + 28 + 5)

    actions = [
        (20, 20, "Use retrieved docs directly"),
        (180, 20, "Re-retrieve with adjusted params"),
        (350, 20, "Fetch from PubMed live"),
    ]
    for x, y, label in actions:
        d.add(String(x + 50, y, label, fontSize=7, fillColor=PALETTE["gray"], textAnchor="middle"))

    d.add(String(240, 160, "CRAG Decision Thresholds", fontSize=11,
                 fillColor=PALETTE["dark"], textAnchor="middle", fontName="Helvetica-Bold"))
    return d


def build_eval_bar_chart():
    d = Drawing(480, 220)
    bc = VerticalBarChart()
    bc.x = 60
    bc.y = 40
    bc.height = 150
    bc.width = 390
    bc.data = [[0.9028, 0.9929, 1.0, 1.0, 1.0, 1.0, 0.7946, 0.8411, 0.9995]]
    bc.categoryAxis.categoryNames = [
        "BERTScore", "Sem.Sim", "Ctx.Prec", "Ctx.Rec",
        "MRR", "NDCG@5", "Ent.Rec", "Faith.", "Relev."
    ]
    bc.categoryAxis.labels.angle = 30
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.labels.dy = -10
    bc.valueAxis.valueMin = 0.7
    bc.valueAxis.valueMax = 1.05
    bc.valueAxis.valueStep = 0.05
    bc.valueAxis.labels.fontSize = 7
    bc.bars[0].fillColor = PALETTE["blue"]
    bc.bars[0].strokeColor = None
    bc.barWidth = 30
    d.add(bc)

    d.add(Line(60, 40 + 150 * (0.90 - 0.7) / 0.35, 450, 40 + 150 * (0.90 - 0.7) / 0.35,
               strokeColor=PALETTE["red"], strokeWidth=1, strokeDashArray=[4, 2]))
    d.add(String(455, 40 + 150 * (0.90 - 0.7) / 0.35 - 3, "target", fontSize=7, fillColor=PALETTE["red"]))

    d.add(String(240, 205, "Evaluation Metrics (DeepSeek Backend)", fontSize=11,
                 fillColor=PALETTE["dark"], textAnchor="middle", fontName="Helvetica-Bold"))
    return d


def build_faithfulness_pie():
    d = Drawing(240, 160)
    pc = Pie()
    pc.x = 50
    pc.y = 20
    pc.width = 120
    pc.height = 120
    pc.data = [65, 35]
    pc.labels = ["Semantic 65%", "Lexical 35%"]
    pc.slices[0].fillColor = PALETTE["blue"]
    pc.slices[1].fillColor = PALETTE["orange"]
    pc.slices[0].strokeColor = PALETTE["white"]
    pc.slices[1].strokeColor = PALETTE["white"]
    pc.slices[0].strokeWidth = 2
    pc.slices[1].strokeWidth = 2
    pc.sideLabels = True
    pc.slices[0].labelRadius = 1.3
    pc.slices[1].labelRadius = 1.3
    pc.slices[0].fontSize = 8
    pc.slices[1].fontSize = 8
    d.add(pc)
    d.add(String(120, 148, "Faithfulness Metric", fontSize=10,
                 fillColor=PALETTE["dark"], textAnchor="middle", fontName="Helvetica-Bold"))
    return d


def build_timing_chart():
    d = Drawing(480, 130)
    total_w = 420
    bar_h = 24
    y_base = 50
    x_start = 60

    stages = [
        ("classify", 4, PALETTE["purple"]),
        ("retrieve", 3195, PALETTE["orange"]),
        ("evaluate", 1, PALETTE["red"]),
        ("generate", 3000, PALETTE["green"]),
    ]
    total_ms = sum(s[1] for s in stages)

    cx = x_start
    for name, ms, color in stages:
        w = max((ms / total_ms) * total_w, 8)
        d.add(Rect(cx, y_base, w, bar_h, fillColor=color, strokeColor=None, rx=3, ry=3))
        if w > 40:
            d.add(String(cx + w / 2, y_base + bar_h / 2 - 3, f"{name}",
                         fontSize=8, fillColor=PALETTE["white"], textAnchor="middle", fontName="Helvetica-Bold"))
            d.add(String(cx + w / 2, y_base - 12, f"{ms}ms" if ms > 1 else "<1ms",
                         fontSize=7, fillColor=PALETTE["dark"], textAnchor="middle"))
        else:
            d.add(String(cx + w / 2, y_base + bar_h + 5, name,
                         fontSize=7, fillColor=PALETTE["dark"], textAnchor="middle"))
            d.add(String(cx + w / 2, y_base - 12, "<1ms",
                         fontSize=7, fillColor=PALETTE["dark"], textAnchor="middle"))
        cx += w + 2

    d.add(String(240, 105, "Pipeline Execution Timeline (sample query)", fontSize=10,
                 fillColor=PALETTE["dark"], textAnchor="middle", fontName="Helvetica-Bold"))
    return d


def build_deployment_diagram():
    d = Drawing(480, 180)
    d.add(Rect(20, 40, 260, 120, fillColor=HexColor("#ecf0f1"), strokeColor=PALETTE["dark"],
               strokeWidth=1.5, rx=8, ry=8))
    d.add(String(150, 145, "Docker Container", fontSize=10,
                 fillColor=PALETTE["dark"], textAnchor="middle", fontName="Helvetica-Bold"))

    _rounded_box(d, 40, 100, 100, 28, PALETTE["green"], "FastAPI :8000")
    _rounded_box(d, 160, 100, 100, 28, PALETTE["blue"], "BGE-M3")
    _rounded_box(d, 40, 55, 100, 28, PALETTE["orange"], "FAISS 3040v")
    _rounded_box(d, 160, 55, 100, 28, PALETTE["purple"], "BM25 Index")

    _rounded_box(d, 340, 110, 120, 28, PALETTE["blue"], "DeepSeek API")
    _rounded_box(d, 340, 60, 120, 28, PALETTE["gray"], "Ollama (opt)")

    _arrow(d, 280, 114, 340, 124)
    _arrow(d, 280, 69, 340, 74)

    d.add(String(240, 10, "Deployment Architecture", fontSize=11,
                 fillColor=PALETTE["dark"], textAnchor="middle", fontName="Helvetica-Bold"))
    return d


# ---- Build Story ----
story = []

# Title
story.append(Paragraph("SIRCA-RAG", title_style))
story.append(Paragraph("Semi-Autonomous RAG for Peruvian Medicinal Plant Knowledge Integration", subtitle_style))
story.append(Paragraph("Technical Report &mdash; SimBig / WAIMLAp 2026", subtitle_style))
story.append(Spacer(1, 12))

# Section 1 — Architecture
story.append(Paragraph("1. Architecture Overview", h1))
story.append(Paragraph("The SIRCA-RAG pipeline implements a Corrective RAG (CRAG) architecture with 4 stages:", body))
story.append(build_architecture_diagram())
story.append(Paragraph("<i>Figure 1 &mdash; End-to-end pipeline flow from query to grounded answer.</i>", caption_style))

story.append(Paragraph("<b>1. Query Classification</b> &mdash; Rule-based classifier (factual / exploratory / comparative) that adjusts hybrid retrieval weights dynamically.", bullet))
story.append(Paragraph("<b>2. Hybrid Retrieval</b> &mdash; BGE-M3 dense embeddings (1024d) + BM25 sparse index, fused via Reciprocal Rank Fusion (RRF), reranked with cross-encoder (ms-marco-MiniLM-L-12-v2).", bullet))
story.append(Paragraph("<b>3. CRAG Evaluation</b> &mdash; Threshold-based decision using pre-computed rerank scores for sub-millisecond evaluation.", bullet))
story.append(Paragraph("<b>4. Grounded Generation</b> &mdash; Chain-of-Thought protocol (Extract &rarr; Verify &rarr; Compose) with strict grounding rules.", bullet))

story.append(Spacer(1, 8))
story.append(Paragraph("1.1 CRAG Decision Logic", h2))
story.append(build_crag_decision_diagram())
story.append(Paragraph("<i>Figure 2 &mdash; CRAG evaluation thresholds and corresponding actions.</i>", caption_style))

# Section 2 — Data Corpus
story.append(Paragraph("2. Data Corpus", h1))
story.append(Paragraph("<b>3,040</b> vectorized chunks from <b>8 Peruvian medicinal plant species</b>.", body))
story.append(Paragraph("<b>Sources:</b> PubMed, CrossRef, PeruNPDB, GBIF, WFO, SciELO", body))
story.append(Paragraph("<b>Embedding model:</b> BAAI/bge-m3 (1024 dimensions)", body))
story.append(Paragraph("<b>Chunk size:</b> 512 tokens, overlap: 64", body))
story.append(Spacer(1, 6))

species_data = [
    ["#", "Species", "Common Name", "Key Compounds"],
    ["1", "Uncaria tomentosa", "Cat's Claw / Una de Gato", "Alkaloids, oxindoles"],
    ["2", "Lepidium meyenii", "Maca", "Macamides, glucosinolates"],
    ["3", "Croton lechleri", "Dragon's Blood", "Taspine, proanthocyanidins"],
    ["4", "Minthostachys mollis", "Muna", "Pulegone, menthone"],
    ["5", "Erythroxylum coca", "Coca", "Cocaine alkaloids, flavonoids"],
    ["6", "Smallanthus sonchifolius", "Yacon", "FOS, phenolic acids"],
    ["7", "Physalis peruviana", "Aguaymanto", "Withanolides, carotenoids"],
    ["8", "Buddleja incana", "Quishuar / Kiswar", "Flavonoids, iridoids"],
]
t = Table(species_data, colWidths=[25, 145, 150, 150])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEADER_BG), ("ALIGN", (0, 0), (0, -1), "CENTER")] + TABLE_BASE))
story.append(t)
story.append(Paragraph("<i>Table 1 &mdash; Target species with representative bioactive compound classes.</i>", caption_style))

# Section 3 — Evaluation Results
story.append(PageBreak())
story.append(Paragraph("3. Evaluation Results (DeepSeek Backend)", h1))

story.append(build_eval_bar_chart())
story.append(Paragraph("<i>Figure 3 &mdash; All 9 evaluation metrics. Red dashed line = BERTScore target (0.90).</i>", caption_style))

eval_data = [
    ["Metric", "Score", "Target", "Status"],
    ["BERTScore F1 (roberta-large)", "0.9028", ">= 0.90", "PASS"],
    ["Semantic Similarity (cross-encoder)", "0.9929", "--", "--"],
    ["Context Precision", "1.0000", "--", "--"],
    ["Context Recall", "1.0000", "--", "--"],
    ["MRR", "1.0000", "--", "--"],
    ["NDCG@5", "1.0000", "--", "--"],
    ["Entity Recall", "0.7946", "--", "--"],
    ["Faithfulness", "0.8411", ">= 0.80", "PASS"],
    ["Answer Relevancy", "0.9995", "--", "--"],
]
t2 = Table(eval_data, colWidths=[220, 80, 80, 60])
t2.setStyle(TableStyle(
    [("BACKGROUND", (0, 0), (-1, 0), HEADER_BG), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
     ("TEXTCOLOR", (3, 1), (3, 1), PASS_COLOR), ("FONTNAME", (3, 1), (3, 1), "Helvetica-Bold"),
     ("TEXTCOLOR", (3, 8), (3, 8), PASS_COLOR), ("FONTNAME", (3, 8), (3, 8), "Helvetica-Bold")]
    + TABLE_BASE
))
story.append(t2)
story.append(Paragraph("<i>Table 2 &mdash; Full evaluation metrics with pass/fail thresholds.</i>", caption_style))

story.append(Paragraph("3.1 Faithfulness Metric Composition", h2))
story.append(build_faithfulness_pie())
story.append(Paragraph("<i>Figure 4 &mdash; Hybrid faithfulness: 65% semantic (cross-encoder) + 35% lexical (word overlap).</i>", caption_style))

# Section 4 — Technical Decisions
story.append(PageBreak())
story.append(Paragraph("4. Key Technical Decisions", h1))
story.append(Paragraph("<b>Chain-of-Thought Grounding:</b> 3-step protocol prevents data invention. The model internally (1) extracts facts from context, (2) verifies each against sources, (3) composes the final answer. Only Step 3 is shown to the user.", body))
story.append(Paragraph("<b>Hybrid Faithfulness Metric:</b> 65% semantic (cross-encoder sentence-level scoring via ms-marco-MiniLM-L-12-v2 with sigmoid normalization) + 35% lexical (content-word overlap with bilingual stopword filtering EN+ES).", body))
story.append(Paragraph("<b>BERTScore with roberta-large:</b> DeBERTa variants caused OverflowError on this system. roberta-large provides stable F1 >= 0.90.", body))
story.append(Paragraph("<b>Temperature 0.0:</b> Deterministic generation for reproducible evaluation across runs.", body))
story.append(Paragraph("<b>Bilingual Support:</b> System prompt instructs model to answer in the same language as the query (Spanish or English). Boilerplate filtering handles both languages.", body))
story.append(Paragraph("<b>Pre-computed Rerank Scores:</b> CRAG evaluation uses scores already computed during the reranking step, enabling sub-millisecond decision making without additional model inference.", body))

decisions_data = [
    ["Decision", "Rationale", "Impact"],
    ["CoT Grounding", "Prevents hallucination via 3-step verify", "Faithfulness 0.84"],
    ["Hybrid Faithfulness", "Catches paraphrases + exact matches", "Robust scoring"],
    ["roberta-large", "DeBERTa OverflowError on system", "Stable F1 >= 0.90"],
    ["Temperature 0.0", "Deterministic for evaluation", "Reproducible"],
    ["Bilingual ES/EN", "Auto-detect query language", "Broader reach"],
    ["Pre-computed scores", "No inference at eval time", "< 1ms evaluate"],
]
t_dec = Table(decisions_data, colWidths=[120, 210, 120])
t_dec.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEADER_BG)] + TABLE_BASE))
story.append(Spacer(1, 8))
story.append(t_dec)
story.append(Paragraph("<i>Table 3 &mdash; Summary of key technical decisions and their impact.</i>", caption_style))

# Section 5 — Pipeline Trace
story.append(Paragraph("5. Pipeline Trace Example", h1))
story.append(Paragraph('Query: "What are the main alkaloids in Uncaria tomentosa?"', body))

story.append(build_timing_chart())
story.append(Paragraph("<i>Figure 5 &mdash; Execution timeline showing relative duration of each pipeline stage.</i>", caption_style))

trace_data = [
    ["Node", "Duration", "Details"],
    ["classify", "< 1ms", "category: exploratory, confidence: 0.60"],
    ["retrieve", "3,195ms", "5 documents, hybrid alpha: 0.6, BGE-M3 + BM25 + rerank"],
    ["evaluate", "< 1ms", "action: accept, confidence: 0.89 (pre-computed scores)"],
    ["generate", "varies", "DeepSeek: ~30-150s | Template: <1ms | Ollama: ~10-30s"],
]
t3 = Table(trace_data, colWidths=[70, 80, 310])
t3.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEADER_BG)] + TABLE_BASE))
story.append(t3)
story.append(Paragraph("<i>Table 4 &mdash; Detailed trace with per-node timing and parameters.</i>", caption_style))

# Section 6 — Web Service
story.append(Paragraph("6. Web Service API", h1))
api_data = [
    ["Method", "Path", "Description"],
    ["GET", "/", "Interactive dark-themed frontend with example queries"],
    ["GET", "/api/health", "System status, backend availability, vectorstore size"],
    ["POST", "/api/query", "Full pipeline: query, backend, language -> answer, citations, trace"],
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
story.append(Paragraph("<i>Table 5 &mdash; REST API endpoints.</i>", caption_style))

story.append(Paragraph("6.1 Request / Response Schema", h2))
schema_data = [
    ["Field", "Type", "Description"],
    ["query", "string", "User question (3-1000 chars)"],
    ["backend", "string", "deepseek | ollama | template"],
    ["language", "string", "auto | es | en"],
]
t5 = Table(schema_data, colWidths=[80, 80, 300])
t5.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEADER_BG)] + TABLE_BASE))
story.append(Paragraph("<b>QueryRequest:</b>", body))
story.append(t5)

resp_data = [
    ["Field", "Type", "Description"],
    ["answer", "string", "Grounded response text"],
    ["citations", "Citation[]", "Source documents with title, authors, DOI, PMID"],
    ["classification", "dict", "Query category + confidence"],
    ["crag_action", "string", "accept | refine | web_search"],
    ["confidence", "float", "CRAG evaluation confidence"],
    ["latency_ms", "int", "End-to-end pipeline latency"],
    ["trace_summary", "string[]", "Per-node execution trace"],
]
t6 = Table(resp_data, colWidths=[90, 80, 290])
t6.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEADER_BG)] + TABLE_BASE))
story.append(Spacer(1, 6))
story.append(Paragraph("<b>QueryResponse:</b>", body))
story.append(t6)
story.append(Paragraph("<i>Tables 6-7 &mdash; API request and response schemas.</i>", caption_style))

# Section 7 — Deployment
story.append(PageBreak())
story.append(Paragraph("7. Deployment (Docker / Dokploy)", h1))

story.append(build_deployment_diagram())
story.append(Paragraph("<i>Figure 6 &mdash; Deployment architecture showing container internals and external service dependencies.</i>", caption_style))

story.append(Paragraph("<b>Image:</b> python:3.11-slim", body))
story.append(Paragraph("<b>Port:</b> 8000", body))
story.append(Paragraph("<b>Health check:</b> GET /api/health (30s interval, 60s start period)", body))
story.append(Spacer(1, 8))

env_data = [
    ["Variable", "Required", "Default", "Description"],
    ["DEEPSEEK_API_KEY", "Yes", "--", "DeepSeek API authentication key"],
    ["OLLAMA_BASE_URL", "No", "http://localhost:11434", "Ollama server URL"],
    ["HOST", "No", "0.0.0.0", "Bind address"],
    ["PORT", "No", "8000", "Service port"],
]
t7 = Table(env_data, colWidths=[120, 60, 150, 130])
t7.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEADER_BG)] + TABLE_BASE))
story.append(t7)
story.append(Paragraph("<i>Table 8 &mdash; Environment variables for deployment configuration.</i>", caption_style))

story.append(Spacer(1, 8))
story.append(Paragraph("docker compose up --build", code_style))

doc.build(story)
print("PDF created: SIRCA_RAG_Report.pdf")
