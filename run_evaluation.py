"""
SIRCA-RAG: Day 6 — Run Evaluation Benchmark
Usage: python run_evaluation.py --backend deepseek
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.benchmark import run_benchmark

backend = sys.argv[sys.argv.index("--backend") + 1] if "--backend" in sys.argv else "template"
results = run_benchmark(backend=backend)
