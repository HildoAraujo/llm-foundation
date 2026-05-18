import json
import yaml
from datetime import datetime, timezone
from typing import List

from src.loader import load_pdf_text
from src.chunker import chunk_text
from src.embedder import embed_text
from src.retriever import retrieve
from src.generator import generate_answer


def load_questions(path: str) -> list[dict]:
    """
    Read the JSON file and return the list of question dicts.
    """

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: str) -> dict:
    """
    Read YAML config and return as dict.
    """

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_hit(retrieved_chunks: list[str], must_contain: list[str]) -> bool:
    must_contain = [kw.lower() for kw in must_contain]
    for chunk in retrieved_chunks:
        chunk_lower = chunk.lower()
        if all(keyword in chunk_lower for keyword in must_contain):
            return True
    return False


def run_eval(config: dict, questions: list[dict], generate: bool = True) -> dict:
    pdf_path = config["pdf_path"]

    text = load_pdf_text(
        pdf_path,
        loader_type=config.get("loader", "pymupdf")
    )

    chunks = chunk_text(
        text=text,
        strategy=config["chunking"]["strategy"],
        size=config["chunking"]["size"],
        overlap=config["chunking"]["overlap"]
    )

    embeddings = embed_text(
        chunks,
        model=config["embedding_model"]
    )

    results = []
    hits = 0

    for q in questions:
        question_id = q["id"]
        question_text = q["question"]
        must_contain = q["must_contain"]
        expected_outcome = q.get("expected_outcome", "hit")

        retrieval_result = retrieve(
            query=question_text,
            chunks=chunks,
            embeddings=embeddings,
            top_k=config["top_k"],
            embedding_model=config["embedding_model"]
        )

        retrieved_chunks = retrieval_result["chunks"]
        retrieved_ids = retrieval_result["chunk_ids"]
        scores = retrieval_result["scores"]

        keyword_found = check_hit(retrieved_chunks, must_contain)

        # For negative questions, a miss is a pass
        passed = (not keyword_found) if expected_outcome == "miss" else keyword_found

        if passed:
            hits += 1

        answer = generate_answer(
            query=question_text,
            context_chunks=retrieved_chunks,
            model=config["generation_model"],
            max_tokens=config["max_tokens"],
            temperature=0
        ) if generate else None

        result = {
            "id": question_id,
            "question": question_text,
            "expected_outcome": expected_outcome,
            "passed": passed,
            "keyword_found": keyword_found,
            "top_score": max(scores) if scores else 0.0,
            "retrieved_chunk_ids": retrieved_ids,
            "answer": answer,
        }

        results.append(result)

    total_questions = len(questions)
    hit_rate = hits / total_questions if total_questions > 0 else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "total_questions": total_questions,
        "hits": hits,
        "hit_rate": hit_rate,
        "results": results,
    }


def format_results_markdown(results: dict, config: dict) -> str:
    """
    Convert evaluation results into markdown report.
    """

    lines = []

    lines.append("# RAG Evaluation Report\n")

    lines.append("## Run Metadata\n")

    lines.append(f"- Timestamp: {results['timestamp']}")
    lines.append(f"- Embedding Model: {config['embedding_model']}")
    lines.append(f"- Chunk Size: {config['chunking']['size']}")
    lines.append(f"- Chunk Overlap: {config['chunking']['overlap']}")
    lines.append(f"- Top K: {config['top_k']}\n")

    lines.append("## Summary\n")

    lines.append(f"- Total Questions: {results['total_questions']}")
    lines.append(f"- Hits: {results['hits']}")
    lines.append(f"- Hit Rate: {results['hit_rate']:.2%}\n")

    lines.append("## Per-Question Results\n")

    for r in results["results"]:
        if r["expected_outcome"] == "miss":
            status = "CORRECT MISS ✅" if r["passed"] else "WRONG HIT ❌"
        else:
            status = "HIT ✅" if r["passed"] else "MISS ❌"
        lines.append(f"### Q{r['id']}: {r['question']}")
        lines.append(f"- **Status:** {status} | **Top Score:** {r['top_score']:.4f}")
        if r.get("answer"):
            lines.append(f"- **Answer:** {r['answer']}\n")
        else:
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import os

    config = load_config("config.yaml")
    questions = load_questions("evals/questions.json")

    results = run_eval(config, questions)

    report = format_results_markdown(results, config)
    print(report)

    os.makedirs("evals", exist_ok=True)
    with open("evals/report.md", "w") as f:
        f.write(report)