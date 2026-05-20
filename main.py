import argparse
import yaml
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import embed_text
from src.generator import generate_answer
from src.retriever import retrieve, retrieve_with_rerank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    text = load_pdf(config["pdf_path"], loader=config.get("loader", "pymupdf"))

    chunks = chunk_text(
        text=text,
        strategy=config["chunking"]["strategy"],
        size=config["chunking"]["size"],
        overlap=config["chunking"]["overlap"],
    )

    embeddings = embed_text(chunks, config["embedding_model"])

    rerank_cfg = config.get("retrieval", {}).get("rerank", {})
    if rerank_cfg.get("enabled", False):
        retrieval_result = retrieve_with_rerank(
            query=args.question,
            chunks=chunks,
            embeddings=embeddings,
            initial_top_k=rerank_cfg["initial_top_k"],
            final_top_k=config["retrieval"]["top_k"],
            rerank_model=rerank_cfg["model"],
            embedding_model=config["embedding_model"],
        )
    else:
        top_k = config.get("retrieval", {}).get("top_k", config.get("top_k", 3))
        retrieval_result = retrieve(
            query=args.question,
            chunks=chunks,
            embeddings=embeddings,
            top_k=top_k,
            embedding_model=config["embedding_model"],
        )

    retrieved_chunks = retrieval_result["chunks"]

    print("\n=== RETRIEVED CHUNKS ===\n")
    for chunk in retrieved_chunks:
        print(chunk)
        print("\n-----------------\n")

    answer = generate_answer(
        query=args.question,
        context_chunks=retrieved_chunks,
        model=config["generation_model"],
        max_tokens=config["max_tokens"],
        temperature=0,
    )

    print("\n=== FINAL ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()

