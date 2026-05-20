"""
SIRCA-RAG Evaluation Runner.

Usage:
  python run_evaluation.py                          # Benchmark with template backend
  python run_evaluation.py --backend deepseek       # Benchmark with DeepSeek
  python run_evaluation.py --ablation               # Full ablation study
  python run_evaluation.py --ablation --configs full dense_only sparse_only
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

parser = argparse.ArgumentParser(description="SIRCA-RAG Evaluation")
parser.add_argument("--backend", default="template", choices=["template", "deepseek", "ollama"])
parser.add_argument("--ablation", action="store_true", help="Run ablation study instead of benchmark")
parser.add_argument("--configs", nargs="*", default=None, help="Ablation configs to run")
args = parser.parse_args()

if args.ablation:
    from evaluation.ablation import run_ablation, print_ablation_table, save_ablation_results, ABLATION_CONFIGS
    configs = ABLATION_CONFIGS
    if args.configs:
        configs = [c for c in ABLATION_CONFIGS if c.name in args.configs]
    results = run_ablation(configs=configs, backend=args.backend)
    print_ablation_table(results)
    save_ablation_results(results, args.backend)
else:
    from evaluation.benchmark import run_benchmark
    results = run_benchmark(backend=args.backend)
