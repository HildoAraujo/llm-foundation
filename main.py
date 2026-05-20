import argparse
import yaml
from src.loader import load_pdf
from src.chunker import chunk_text
from src.embedder import embed_text
from src.bm25_retriever import build_bm25_index
from src.retriever import retrieve, retrieve_with_rerank
from src.hybrid_retriever import hybrid_retrieve, hybrid_retrieve_rerank
from src.generator import generate_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    retrieval_cfg = config.get("retrieval", {})
    strategy = retrieval_cfg.get("strategy", "dense")
    top_k = retrieval_cfg.get("top_k", 3)
    initial_top_k = retrieval_cfg.get("initial_top_k", 20)
    rerank_model = retrieval_cfg.get("rerank_model", "BAAI/bge-reranker-base")
    embedding_model = config["embedding_model"]

    text = load_pdf(config["pdf_path"], loader=config.get("loader", "pymupdf"))
    chunks = chunk_text(
        text=text,
        strategy=config["chunking"]["strategy"],
        size=config["chunking"]["size"],
        overlap=config["chunking"]["overlap"],
    )

    embeddings = embed_text(chunks, embedding_model) if strategy != "bm25_only" else None
    bm25_index = build_bm25_index(chunks) if strategy in ("bm25_only", "hybrid", "hybrid_rerank") else None

    if strategy == "dense":
        assert embeddings is not None
        retrieval_result = retrieve(
            query=args.question, chunks=chunks, embeddings=embeddings,
            top_k=top_k, embedding_model=embedding_model,
        )
    elif strategy == "dense_rerank":
        assert embeddings is not None
        retrieval_result = retrieve_with_rerank(
            query=args.question, chunks=chunks, embeddings=embeddings,
            initial_top_k=initial_top_k, final_top_k=top_k,
            rerank_model=rerank_model, embedding_model=embedding_model,
        )
    elif strategy == "bm25_only":
        from src.bm25_retriever import bm25_retrieve
        assert bm25_index is not None
        indices, scores = bm25_retrieve(args.question, chunks, bm25_index, top_k=top_k)
        retrieval_result = {"chunks": [chunks[i] for i in indices], "chunk_ids": indices, "scores": scores}
    elif strategy == "hybrid":
        assert embeddings is not None and bm25_index is not None
        retrieval_result = hybrid_retrieve(
            query=args.question, chunks=chunks, embeddings=embeddings,
            bm25_index=bm25_index, initial_top_k=initial_top_k, final_top_k=top_k,
            embedding_model=embedding_model,
        )
    elif strategy == "hybrid_rerank":
        assert embeddings is not None and bm25_index is not None
        retrieval_result = hybrid_retrieve_rerank(
            query=args.question, chunks=chunks, embeddings=embeddings,
            bm25_index=bm25_index, initial_top_k=initial_top_k, final_top_k=top_k,
            embedding_model=embedding_model, rerank_model=rerank_model,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

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

