import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.run_eval import load_questions, run_eval

_BASE = {
    "pdf_path": "data/Identifying and Scaling AI Use Cases.pdf",
    "loader": "pymupdf",
    "chunking": {"strategy": "fixed", "size": 500, "overlap": 50},
    "embedding_model": "text-embedding-3-small",
    "generation_model": "claude-sonnet-4-6",
    "max_tokens": 500,
    "initial_top_k": 20,
    "rerank_model": "BAAI/bge-reranker-base",
}

CONFIGS = [
    # Day 3/4 baselines (dense only)
    {**_BASE, "name": "dense_k3",      "strategy": "dense",        "top_k": 3},
    {**_BASE, "name": "dense_k6",      "strategy": "dense",        "top_k": 6},
    {**_BASE, "name": "dense_rerank",  "strategy": "dense_rerank", "top_k": 3},
    # Day 5 — BM25 and hybrid
    {**_BASE, "name": "bm25_only",     "strategy": "bm25_only",    "top_k": 3},
    {**_BASE, "name": "hybrid_k3",     "strategy": "hybrid",       "top_k": 3},
    {**_BASE, "name": "hybrid_k6",     "strategy": "hybrid",       "top_k": 6},
    {**_BASE, "name": "hybrid_rerank", "strategy": "hybrid_rerank","top_k": 3},
]


def run_comparison(questions_path: str = "evals/questions.json") -> list[dict]:
    questions = load_questions(questions_path)
    rows = []

    print(f"\n{'Config':<20} {'Strategy':<16} {'Top K':>6} {'Hits':>8} {'Hit Rate':>10} {'Avg Score':>11}")
    print("-" * 78)

    for cfg in CONFIGS:
        results = run_eval(cfg, questions, generate=False)
        scores = [r["top_score"] for r in results["results"]]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        row = {
            "name": cfg["name"],
            "strategy": cfg.get("strategy", "dense"),
            "chunk_size": cfg["chunking"]["size"],
            "overlap": cfg["chunking"]["overlap"],
            "top_k": cfg["top_k"],
            "hits": results["hits"],
            "total": results["total_questions"],
            "hit_rate": results["hit_rate"],
            "avg_score": avg_score,
        }
        rows.append(row)

        print(
            f"{cfg['name']:<20} {cfg.get('strategy', 'dense'):<16} {cfg['top_k']:>6} "
            f"{results['hits']:>4}/{results['total_questions']:<3} "
            f"{results['hit_rate']:>9.0%} {avg_score:>10.4f}"
        )

    best = max(rows, key=lambda r: (r["hit_rate"], r["avg_score"]))
    worst = min(rows, key=lambda r: (r["hit_rate"], r["avg_score"]))
    print(f"\nWinner: {best['name']} ({best['hit_rate']:.0%} hit rate, avg score {best['avg_score']:.4f})")
    print(f"Worst:  {worst['name']} ({worst['hit_rate']:.0%} hit rate, avg score {worst['avg_score']:.4f})")

    return rows


def write_markdown(rows: list[dict], path: str = "evals/results/comparison.md") -> None:
    lines = ["# Config Comparison\n"]
    lines.append(f"Questions: {rows[0]['total']} | Embedding: text-embedding-3-small | Loader: pymupdf\n")
    lines.append("| Config | Strategy | Top K | Hit Rate | Avg Top Score |")
    lines.append("|--------|----------|-------|----------|---------------|")

    for r in rows:
        lines.append(
            f"| {r['name']} | {r['strategy']} | {r['top_k']} "
            f"| {r['hit_rate']:.0%} | {r['avg_score']:.4f} |"
        )

    best = max(rows, key=lambda r: (r["hit_rate"], r["avg_score"]))
    worst = min(rows, key=lambda r: (r["hit_rate"], r["avg_score"]))
    lines.append(f"\n**Winner:** `{best['name']}` — {best['hit_rate']:.0%} hit rate, avg score {best['avg_score']:.4f}")
    lines.append(f"**Worst:** `{worst['name']}` — {worst['hit_rate']:.0%} hit rate, avg score {worst['avg_score']:.4f}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    rows = run_comparison()
    write_markdown(rows)
