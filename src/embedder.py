import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_text(texts: list[str], model:str) -> np.ndarray:

    response = client.embeddings.create(
        input = texts,
        model = model
    )

    embeddings = []

    for item in response.data:
        embeddings.append(item.embedding)

    return np.array(embeddings)  

if __name__ == "__main__":
    result = embed_text(["hello", "world"], "text-embedding-3-small")
    print(result)
    