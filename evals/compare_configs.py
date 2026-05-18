import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.run_eval import load_questions, run_eval

questions = load_questions("evals/questions.json")

configs = [
    {
        "label": "A — top_k=3, chunk=500, pymupdf",
        "pdf_path": "data/Identifying and Scaling AI Use Cases.pdf",
        "loader": "pymupdf",
        "chunking": {"strategy": "fixed", "size": 500, "overlap": 50},
        "embedding_model": "text-embedding-3-small",
        "top_k": 3,
        "generation_model": "claude-sonnet-4-6",
        "max_tokens": 500,
    },
    {
        "label": "B — top_k=6, chunk=500, pymupdf",
        "pdf_path": "data/Identifying and Scaling AI Use Cases.pdf",
        "loader": "pymupdf",
        "chunking": {"strategy": "fixed", "size": 500, "overlap": 50},
        "embedding_model": "text-embedding-3-small",
        "top_k": 6,
        "generation_model": "claude-sonnet-4-6",
        "max_tokens": 500,
    },
    {
        "label": "C — top_k=3, chunk=1000, pymupdf",
        "pdf_path": "data/Identifying and Scaling AI Use Cases.pdf",
        "loader": "pymupdf",
        "chunking": {"strategy": "fixed", "size": 1000, "overlap": 100},
        "embedding_model": "text-embedding-3-small",
        "top_k": 3,
        "generation_model": "claude-sonnet-4-6",
        "max_tokens": 500,
    },
    {
        "label": "D — top_k=6, chunk=500, pdfplumber",
        "pdf_path": "data/Identifying and Scaling AI Use Cases.pdf",
        "loader": "pdfplumber",
        "chunking": {"strategy": "fixed", "size": 500, "overlap": 50},
        "embedding_model": "text-embedding-3-small",
        "top_k": 6,
        "generation_model": "claude-sonnet-4-6",
        "max_tokens": 500,
    },
]

print(f"\n{'Config':<45} {'Hits':>6} {'Hit Rate':>10} {'Avg Score':>11}")
print("-" * 76)

all_results = []

for cfg in configs:
    results = run_eval(cfg, questions)
    scores = [r["top_score"] for r in results["results"]]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"{cfg['label']:<45} {results['hits']:>6}/{results['total_questions']} {results['hit_rate']:>9.0%} {avg_score:>10.4f}")
    all_results.append((cfg["label"], results))

print()
