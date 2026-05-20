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
}

_RERANK_MODEL = "BAAI/bge-reranker-base"

CONFIGS = [
    # Day 3 baselines
    {**_BASE, "name": "baseline",        "top_k": 3, "rerank": {"enabled": False}},
    {**_BASE, "name": "wider_retrieval", "top_k": 6, "rerank": {"enabled": False}},
    # Day 4 — reranker variants
    {**_BASE, "name": "rerank_k3_from_20", "top_k": 3,
     "rerank": {"enabled": True, "model": _RERANK_MODEL, "initial_top_k": 20}},
    {**_BASE, "name": "rerank_k6_from_20", "top_k": 6,
     "rerank": {"enabled": True, "model": _RERANK_MODEL, "initial_top_k": 20}},
    {**_BASE, "name": "rerank_k3_from_50", "top_k": 3,
     "rerank": {"enabled": True, "model": _RERANK_MODEL, "initial_top_k": 50}},
]


def run_comparison(questions_path: str = "evals/questions.json") -> list[dict]:
    questions = load_questions(questions_path)
    rows = []

    print(f"\n{'Config':<20} {'Chunk':>6} {'Overlap':>8} {'Top K':>6} {'Hits':>8} {'Hit Rate':>10} {'Avg Score':>11}")
    print("-" * 75)

    for cfg in CONFIGS:
        results = run_eval(cfg, questions, generate=False)
        scores = [r["top_score"] for r in results["results"]]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        row = {
            "name": cfg["name"],
            "chunk_size": cfg["chunking"]["size"],
            "overlap": cfg["chunking"]["overlap"],
            "top_k": cfg["top_k"],
            "rerank": cfg.get("rerank", {}).get("enabled", False),
            "hits": results["hits"],
            "total": results["total_questions"],
            "hit_rate": results["hit_rate"],
            "avg_score": avg_score,
        }
        rows.append(row)

        print(
            f"{cfg['name']:<20} {cfg['chunking']['size']:>6} {cfg['chunking']['overlap']:>8} "
            f"{cfg['top_k']:>6} {results['hits']:>4}/{results['total_questions']:<3} "
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
    lines.append("| Config | Chunk | Overlap | Top K | Rerank | Hit Rate | Avg Top Score |")
    lines.append("|--------|-------|---------|-------|--------|----------|---------------|")

    for r in rows:
        rerank_flag = "yes" if r.get("rerank") else "no"
        lines.append(
            f"| {r['name']} | {r['chunk_size']} | {r['overlap']} | {r['top_k']} "
            f"| {rerank_flag} | {r['hit_rate']:.0%} | {r['avg_score']:.4f} |"
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
