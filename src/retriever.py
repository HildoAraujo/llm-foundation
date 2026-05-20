import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import embed_text
from src.reranker import rerank


def retrieve(query: str, chunks: list[str], embeddings: np.ndarray, top_k: int, embedding_model: str) -> dict:
    query_embedding = embed_text(
        [query],
        embedding_model
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(-similarities)[:top_k]

    return {
        "chunks": [chunks[i] for i in top_indices],
        "chunk_ids": [int(i) for i in top_indices],
        "scores": [float(similarities[i]) for i in top_indices],
    }


def retrieve_with_rerank(
    query: str,
    chunks: list[str],
    embeddings: np.ndarray,
    initial_top_k: int,
    final_top_k: int,
    rerank_model: str,
    embedding_model: str,
) -> dict:
    candidates = retrieve(query, chunks, embeddings, top_k=initial_top_k, embedding_model=embedding_model)
    reranked_chunks, reranked_scores = rerank(
        query=query,
        chunks=candidates["chunks"],
        model_name=rerank_model,
        top_k=final_top_k,
    )
    return {
        "chunks": reranked_chunks,
        "chunk_ids": [],
        "scores": reranked_scores,
    }


if __name__ == "__main__":
    chunks = [
        "Python is a programming language",
        "Dogs are animals",
        "Machine learning uses data"
    ]

    embeddings = embed_text(chunks, "text-embedding-3-small")

    results = retrieve(
        query="what is Python",
        chunks=chunks,
        embeddings=embeddings,
        top_k=1,
        embedding_model="text-embedding-3-small"
    )

    print(results)