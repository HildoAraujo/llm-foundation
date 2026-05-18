import os 
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_answer(query: str, context_chunks: list[str], model:str, max_tokens: int, temperature: float) -> str:
    context = "\n\n".join(context_chunks)

    system_prompt = f"""
    You are a helpful assistant.

    Answer the user's question using ONLY the context below.
    
    If the answer is not in the context, say:
    "I could not find thr answer in the provided context."

    CONTEXT:
    {context}
    """

    response = client.messages.create(
        model=model,
        max_tokens = max_tokens,
        system = system_prompt,
        messages = [
            {
                "role": "user",
                "content": query
            }
        ] 

    )
    return response.content[0].text


chunks = [
    "Python is a programming language",
    "It is commonly used in Ai"
]

answer = generate_answer (
    query = "Whats is python used for?",
    context_chunks = chunks,
    model = "claude-sonnet-4-6",
    max_tokens = 200, 
    temperature = 0
)

print(answer)