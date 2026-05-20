import numpy as np
from rank_bm25 import BM25Okapi

from src.bm25_retriever import bm25_retrieve
from src.reranker import rerank
from src.retriever import retrieve


def reciprocal_rank_fusion(
    dense_ranked_indices: list[int],
    bm25_ranked_indices: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranked_indices):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(
    query: str,
    chunks: list[str],
    embeddings: np.ndarray,
    bm25_index: BM25Okapi,
    initial_top_k: int,
    final_top_k: int,
    embedding_model: str,
) -> dict:
    dense_result = retrieve(query, chunks, embeddings, top_k=initial_top_k, embedding_model=embedding_model)
    dense_indices = dense_result["chunk_ids"]

    bm25_indices, _ = bm25_retrieve(query, chunks, bm25_index, top_k=initial_top_k)

    fused = reciprocal_rank_fusion(dense_indices, bm25_indices)[:final_top_k]
    return {
        "chunks": [chunks[i] for i, _ in fused],
        "chunk_ids": [i for i, _ in fused],
        "scores": [s for _, s in fused],
    }


def hybrid_retrieve_rerank(
    query: str,
    chunks: list[str],
    embeddings: np.ndarray,
    bm25_index: BM25Okapi,
    initial_top_k: int,
    final_top_k: int,
    embedding_model: str,
    rerank_model: str,
) -> dict:
    # Hybrid fusion to get initial_top_k candidates, then cross-encoder rerank to final_top_k
    candidates = hybrid_retrieve(
        query, chunks, embeddings, bm25_index,
        initial_top_k=initial_top_k, final_top_k=initial_top_k,
        embedding_model=embedding_model,
    )
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
