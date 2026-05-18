import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import embed_text

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