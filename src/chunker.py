
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


text = "dskndskn"
result = chunk_text(
    text=text,
    size = 4,
    strategy = "fixed",
    overlap =1
) 
print(result)       