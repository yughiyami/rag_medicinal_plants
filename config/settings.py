import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = None  # Optional: set for 10 req/s instead of 3

TARGET_SPECIES = [
    "Uncaria tomentosa",
    "Lepidium meyenii",
    "Croton lechleri",
    "Minthostachys mollis",
    "Erythroxylum coca",
    "Smallanthus sonchifolius",
    "Physalis peruviana",
    "Buddleja incana",
]

SEARCH_QUERIES = [
    '"{species}"[Title/Abstract] AND (medicinal OR pharmacological OR therapeutic)',
    '"{species}"[Title/Abstract] AND (bioactive OR phytochemical OR ethnobotanical)',
]

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", "]

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5
HYBRID_ALPHA = 0.6  # weight for dense retrieval (1-alpha for BM25)

GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

OLLAMA_MODEL = "qwen3.5:latest"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
