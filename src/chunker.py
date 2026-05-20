_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(text: str, strategy: str, size: int = 500, overlap: int = 50, **kwargs) -> list[str]:
    if strategy == "fixed":
        return _chunk_fixed(text, size, overlap)
    elif strategy == "recursive":
        return chunk_text_recursive(text, size, overlap)
    elif strategy == "semantic":
        from src.semantic_chunker import chunk_text_semantic
        return chunk_text_semantic(
            text=text,
            embedding_model=kwargs.get("embedding_model", "text-embedding-3-small"),
            breakpoint_percentile=kwargs.get("breakpoint_percentile", 90.0),
            max_chunk_size=kwargs.get("max_chunk_size", 2000),
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def _chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i: i + size])
        i += size - overlap
    return chunks


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    """Split text into pieces all <= size, using seps as fallback separators."""
    if len(text) <= size:
        return [text] if text.strip() else []
    if not seps:
        return [text[i: i + size] for i in range(0, len(text), size)]

    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _recursive_split(text, size, rest)

    pieces = []
    for part in text.split(sep):
        part = part.strip()
        if not part:
            continue
        if len(part) <= size:
            pieces.append(part)
        else:
            pieces.extend(_recursive_split(part, size, rest))
    return pieces


def _merge_with_overlap(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge pieces into chunks <= size, seeding each new chunk with overlap."""
    if not pieces:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        sep = 1 if current else 0
        if current_len + sep + len(piece) > size and current:
            chunks.append(" ".join(current))
            if overlap > 0:
                tail = chunks[-1][-overlap:]
                if len(tail) + 1 + len(piece) <= size:
                    current = [tail, piece]
                    current_len = len(tail) + 1 + len(piece)
                else:
                    current = [piece]
                    current_len = len(piece)
            else:
                current = [piece]
                current_len = len(piece)
        else:
            current.append(piece)
            current_len += sep + len(piece)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def chunk_text_recursive(text: str, size: int, overlap: int) -> list[str]:
    pieces = _recursive_split(text, size, _SEPARATORS)
    return _merge_with_overlap(pieces, size, overlap)


if __name__ == "__main__":
    sample = "First paragraph here.\n\nSecond paragraph with more content here.\n\nThird short one."
    for c in chunk_text_recursive(sample, size=50, overlap=10):
        print(repr(c))
        print("---")
