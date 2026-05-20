import json
import yaml
from datetime import datetime, timezone

from src.loader import load_pdf_text
from src.chunker import chunk_text
from src.embedder import embed_text
from src.retriever import retrieve, retrieve_with_rerank
from src.bm25_retriever import build_bm25_index, bm25_retrieve
from src.hybrid_retriever import hybrid_retrieve, hybrid_retrieve_rerank
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


def _get_strategy(config: dict) -> str:
    if "strategy" in config:
        return config["strategy"]
    # Legacy fallback for Day 3/4 configs using rerank dict
    if config.get("rerank", {}).get("enabled"):
        return "dense_rerank"
    return "dense"


def run_eval(config: dict, questions: list[dict], generate: bool = True) -> dict:
    pdf_path = config["pdf_path"]
    strategy = _get_strategy(config)
    top_k = config["top_k"]
    initial_top_k = config.get("initial_top_k", 20)
    rerank_model = config.get("rerank_model", "BAAI/bge-reranker-base")

    text = load_pdf_text(pdf_path, loader_type=config.get("loader", "pymupdf"))

    chunks = chunk_text(
        text=text,
        strategy=config["chunking"]["strategy"],
        size=config["chunking"]["size"],
        overlap=config["chunking"]["overlap"],
    )

    embeddings = None
    if strategy != "bm25_only":
        embeddings = embed_text(chunks, model=config["embedding_model"])

    bm25_index = None
    if strategy in ("bm25_only", "hybrid", "hybrid_rerank"):
        bm25_index = build_bm25_index(chunks)

    results = []
    hits = 0

    for q in questions:
        question_id = q["id"]
        question_text = q["question"]
        must_contain = q["must_contain"]
        expected_outcome = q.get("expected_outcome", "hit")

        if strategy == "dense":
            retrieval_result = retrieve(
                query=question_text, chunks=chunks, embeddings=embeddings,
                top_k=top_k, embedding_model=config["embedding_model"],
            )
        elif strategy == "dense_rerank":
            legacy_rerank = config.get("rerank", {})
            retrieval_result = retrieve_with_rerank(
                query=question_text, chunks=chunks, embeddings=embeddings,
                initial_top_k=legacy_rerank.get("initial_top_k", initial_top_k),
                final_top_k=top_k,
                rerank_model=legacy_rerank.get("model", rerank_model),
                embedding_model=config["embedding_model"],
            )
        elif strategy == "bm25_only":
            indices, scores = bm25_retrieve(question_text, chunks, bm25_index, top_k=top_k)
            retrieval_result = {
                "chunks": [chunks[i] for i in indices],
                "chunk_ids": indices,
                "scores": scores,
            }
        elif strategy == "hybrid":
            retrieval_result = hybrid_retrieve(
                query=question_text, chunks=chunks, embeddings=embeddings,
                bm25_index=bm25_index, initial_top_k=initial_top_k, final_top_k=top_k,
                embedding_model=config["embedding_model"],
            )
        elif strategy == "hybrid_rerank":
            retrieval_result = hybrid_retrieve_rerank(
                query=question_text, chunks=chunks, embeddings=embeddings,
                bm25_index=bm25_index, initial_top_k=initial_top_k, final_top_k=top_k,
                embedding_model=config["embedding_model"], rerank_model=rerank_model,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

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