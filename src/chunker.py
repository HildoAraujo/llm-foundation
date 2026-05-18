
def chunk_text(text: str, strategy: str, size: int, overlap:int) -> list[str]:
    if strategy == "fixed":

        chunks = []

        i = 0

        while i < len(text):
            chunk = text[i : i + size]
            chunks.append(chunk)
            i += size - overlap
        return chunks
        
    elif strategy == "recursive":
        raise NotImplementedError("recurvise chunking not yet implemented")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")    


if __name__ == "__main__":
    result = chunk_text(text="dskndskn", strategy="fixed", size=4, overlap=1)
    print(result)