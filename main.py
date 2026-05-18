import argparse
import yaml
from pypdf import PdfReader
from src.chunker import chunk_text
from src.embedder import embed_text
from src.generator import generate_answer
from src.retriever import retrieve


def load_pdf (path: str) -> str:
     reader = PdfReader(path)

     text = " "

     for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

     return text

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "question", 
        type = str
    )     

    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    text = load_pdf(config["pdf_path"])

    chunks = chunk_text(
        text = text,
        strategy = config["chunking"]["strategy"],
        size = config["chunking"]["size"],
        overlap = config["chunking"]["overlap"]
    )    

    embeddings = embed_text (
        chunks,
        config["embedding_model"]
    )

    retrieved_chunks = retrieve (
        query = args.question,
        chunks = chunks,
        embeddings = embeddings,
        top_k = config["top_k"],
        embedding_model = config["embedding_model"]
    )

    print("\n=== RETRIEVED CHUNKS ===\n")

    for chunk in retrieved_chunks:
        print(chunk)
        print("\n-----------------\n")

    answer = generate_answer(
        query = args.question,
        context_chunks = retrieved_chunks,
        model = config["generation_model"],
        max_tokens = config["max_tokens"],
        temperature = 0
    )    

    print("\n=== FINAL ANSWER ===\n")
    print(answer)

if __name__ == "__main__":
    main()    
