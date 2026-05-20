import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.run_eval import load_questions, run_eval

_BASE = {
    "pdf_path": "data/Identifying and Scaling AI Use Cases.pdf",
    "loader": "pymupdf",
    "embedding_model": "text-embedding-3-small",
    "generation_model": "claude-sonnet-4-6",
    "max_tokens": 500,
    "initial_top_k": 20,
    "rerank_model": "BAAI/bge-reranker-base",
}

_FIXED   = {"strategy": "fixed",     "size": 500, "overlap": 50}
_RECUR   = {"strategy": "recursive", "size": 500, "overlap": 50}
_SEMANT  = {"strategy": "semantic",  "max_chunk_size": 2000, "breakpoint_percentile": 90.0}

CONFIGS = [
    # Days 3-5 baselines (fixed chunking)
    {**_BASE, "name": "fixed_dense",   "chunking": _FIXED,  "strategy": "dense",   "top_k": 3},
    {**_BASE, "name": "fixed_dense_k6","chunking": _FIXED,  "strategy": "dense",   "top_k": 6},
    {**_BASE, "name": "fixed_hybrid",  "chunking": _FIXED,  "strategy": "hybrid",  "top_k": 3},
    # Day 6 — recursive chunking
    {**_BASE, "name": "recur_dense",   "chunking": _RECUR,  "strategy": "dense",   "top_k": 3},
    {**_BASE, "name": "recur_hybrid",  "chunking": _RECUR,  "strategy": "hybrid",  "top_k": 3},
    # Day 6 — semantic chunking
    {**_BASE, "name": "sem_dense",     "chunking": _SEMANT, "strategy": "dense",   "top_k": 3},
    {**_BASE, "name": "sem_hybrid",    "chunking": _SEMANT, "strategy": "hybrid",  "top_k": 3},
]


def run_comparison(
    questions_path: str = "evals/questions.json",
) -> tuple[list[dict], list[dict], dict[int, dict[str, bool]]]:
    questions = load_questions(questions_path)
    rows = []
    # q_pass[question_id][config_name] = passed
    q_pass: dict[int, dict[str, bool]] = {q["id"]: {} for q in questions}

    print(f"\n{'Config':<20} {'Chunking':<12} {'Retrieval':<16} {'Top K':>6} {'Hits':>8} {'Hit Rate':>10} {'Avg Score':>11}")
    print("-" * 88)

    for cfg in CONFIGS:
        results = run_eval(cfg, questions, generate=False)
        scores = [r["top_score"] for r in results["results"]]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        for r in results["results"]:
            q_pass[r["id"]][cfg["name"]] = r["passed"]

        chunking_strat = cfg["chunking"]["strategy"]
        row = {
            "name": cfg["name"],
            "chunking": chunking_strat,
            "strategy": cfg.get("strategy", "dense"),
            "chunk_size": cfg["chunking"].get("size", "—"),
            "top_k": cfg["top_k"],
            "hits": results["hits"],
            "total": results["total_questions"],
            "hit_rate": results["hit_rate"],
            "avg_score": avg_score,
        }
        rows.append(row)

        print(
            f"{cfg['name']:<20} {chunking_strat:<12} {cfg.get('strategy', 'dense'):<16} {cfg['top_k']:>6} "
            f"{results['hits']:>4}/{results['total_questions']:<3} "
            f"{results['hit_rate']:>9.0%} {avg_score:>10.4f}"
        )

    best = max(rows, key=lambda r: (r["hit_rate"], r["avg_score"]))
    worst = min(rows, key=lambda r: (r["hit_rate"], r["avg_score"]))
    print(f"\nWinner: {best['name']} ({best['hit_rate']:.0%} hit rate, avg score {best['avg_score']:.4f})")
    print(f"Worst:  {worst['name']} ({worst['hit_rate']:.0%} hit rate, avg score {worst['avg_score']:.4f})")

    return rows, questions, q_pass


def print_breakdown(
    questions: list[dict], q_pass: dict[int, dict[str, bool]], config_names: list[str]
) -> None:
    col_w = 10
    q_col = 52
    print(f"\n{'Question':<{q_col}}", end="")
    for name in config_names:
        print(f"{name[:col_w]:^{col_w}}", end="")
    print(f"{'Passes':>8}")
    print("-" * (q_col + col_w * len(config_names) + 8))

    for q in questions:
        qid = q["id"]
        label = f"Q{qid}: {q['question']}"[:q_col - 1]
        passes = q_pass[qid]
        total_pass = sum(passes.values())
        print(f"{label:<{q_col}}", end="")
        for name in config_names:
            cell = "✓" if passes.get(name) else "✗"
            print(f"{cell:^{col_w}}", end="")
        print(f"{total_pass:>{8}}/{len(config_names)}")


def write_markdown(
    rows: list[dict],
    questions: list[dict],
    q_pass: dict[int, dict[str, bool]],
    path: str = "evals/results/comparison.md",
) -> None:
    config_names = [r["name"] for r in rows]
    lines = ["# Config Comparison\n"]
    lines.append(f"Questions: {rows[0]['total']} | Embedding: text-embedding-3-small | Loader: pymupdf\n")

    # Summary table
    lines.append("| Config | Chunking | Retrieval | Top K | Hit Rate | Avg Top Score |")
    lines.append("|--------|----------|-----------|-------|----------|---------------|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['chunking']} | {r['strategy']} | {r['top_k']} "
            f"| {r['hit_rate']:.0%} | {r['avg_score']:.4f} |"
        )

    best = max(rows, key=lambda r: (r["hit_rate"], r["avg_score"]))
    worst = min(rows, key=lambda r: (r["hit_rate"], r["avg_score"]))
    lines.append(f"\n**Winner:** `{best['name']}` — {best['hit_rate']:.0%} hit rate, avg score {best['avg_score']:.4f}")
    lines.append(f"**Worst:** `{worst['name']}` — {worst['hit_rate']:.0%} hit rate, avg score {worst['avg_score']:.4f}")

    # Per-question breakdown
    lines.append("\n## Per-Question Breakdown\n")
    header = "| Question | Difficulty |" + "".join(f" {n} |" for n in config_names) + " Passes |"
    sep = "|---|---|" + "".join("---|" for _ in config_names) + "---|"
    lines.append(header)
    lines.append(sep)

    for q in questions:
        qid = q["id"]
        passes = q_pass[qid]
        total_pass = sum(passes.values())
        cells = "".join(f" {'✓' if passes.get(n) else '✗'} |" for n in config_names)
        label = q["question"][:55]
        lines.append(f"| Q{qid}: {label} | {q.get('difficulty', '')} |{cells} {total_pass}/{len(config_names)} |")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    rows, questions, q_pass = run_comparison()
    config_names = [r["name"] for r in rows]
    print_breakdown(questions, q_pass, config_names)
    write_markdown(rows, questions, q_pass)
