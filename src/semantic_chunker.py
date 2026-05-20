import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embedder import embed_text


def _sentences(text: str) -> list[str]:
    import nltk  # runtime import — installed in Anaconda, not in .venv
    try:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def chunk_text_semantic(
    text: str,
    embedding_model: str,
    breakpoint_percentile: float = 90.0,
    max_chunk_size: int = 2000,
) -> list[str]:
    sentences = _sentences(text)
    if len(sentences) <= 1:
        return [text]

    embeddings = embed_text(sentences, embedding_model)

    # Distance between adjacent sentences (high = topic shift)
    distances = [
        1.0 - float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
        for i in range(len(sentences) - 1)
    ]

    # Split where distance exceeds the percentile threshold.
    # Percentile-based threshold makes this adaptive per document — what
    # counts as a "big shift" varies across corpora.
    threshold = float(np.percentile(distances, breakpoint_percentile))
    split_points = {i + 1 for i, d in enumerate(distances) if d > threshold}

    chunks: list[str] = []
    current: list[str] = []
    for i, sentence in enumerate(sentences):
        if i in split_points and current:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))

    # Enforce max_chunk_size — a uniformly-topiced doc would otherwise
    # produce one giant chunk if all distances are below the threshold.
    if max_chunk_size:
        from src.chunker import chunk_text_recursive  # type: ignore[attr-defined]
        final: list[str] = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                final.extend(chunk_text_recursive(chunk, max_chunk_size, overlap=0))
            else:
                final.append(chunk)
        return [c for c in final if c.strip()]

    return [c for c in chunks if c.strip()]
