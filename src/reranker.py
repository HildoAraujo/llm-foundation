from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(query: str, chunks: list[str], model_name: str, top_k: int) -> tuple[list[str], list[float]]:
    model = _get_model(model_name)
    pairs = [(query, chunk) for chunk in chunks]
    scores = model.predict(pairs).tolist()
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:top_k]
    return [c for c, _ in top], [s for _, s in top]
