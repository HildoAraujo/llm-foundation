import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import embed_text

def retrieve(query: str, chunks: list[str], embeddings: np.ndarray, top_k: int, embedding_model: str) -> list [str]:
    query_embedding = embed_text(
        [query],
        embedding_model
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(-similarities)[:top_k]

    results = []

    for idx in top_indices:
        results.append(chunks[idx])

    return results     


chunks = [
    "Python is a programming language",
    "Dogs are animals",
    "Machine learning uses date"
]

embeddings = embed_text (
    chunks,
    "text-embedding-3-small"
)

results = retrieve(
    query = "whats is Python",
    chunks = chunks,
    embeddings = embeddings,
    top_k = 1, 
    embedding_model= "text-embedding-3-small"
)

print (results)