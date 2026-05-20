import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(chunk) for chunk in chunks])


def bm25_retrieve(
    query: str, chunks: list[str], bm25_index: BM25Okapi, top_k: int
) -> tuple[list[int], list[float]]:
    scores = bm25_index.get_scores(tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return top_indices, [float(scores[i]) for i in top_indices]
