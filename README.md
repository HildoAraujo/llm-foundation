# LLM Foundations — RAG Pipeline

A retrieval-augmented generation (RAG) pipeline built from scratch to understand the core mechanics of how LLMs work in production.

## What it does

Ask a question about a PDF and get a grounded answer — no hallucination, only what's in the document.

```
PDF → Chunk → Embed → Retrieve → Generate → Answer
```

## Stack

- **Chunker** — splits raw PDF text into fixed-size overlapping chunks
- **Embedder** — converts chunks to vector embeddings via OpenAI (`text-embedding-3-small`)
- **Retriever** — finds the most semantically relevant chunks using cosine similarity
- **Generator** — sends retrieved context + query to Claude (`claude-sonnet-4-6`) and returns an answer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

Add a PDF to the `data/` folder and update `config.yaml`:

```yaml
pdf_path: "data/your-document.pdf"
chunking:
  strategy: "fixed"
  size: 500
  overlap: 50
embedding_model: "text-embedding-3-small"
top_k: 6
generation_model: "claude-sonnet-4-6"
max_tokens: 500
```

## Usage

```bash
python main.py "Your question here"
```

## What I learned

- How RAG works end-to-end — embeddings, similarity search, prompt injection
- Why chunking strategy and `top_k` directly affect answer quality
- How cosine similarity is surprisingly robust even with noisy PDF text
- How a grounded system prompt prevents hallucination

## What's next

- Swap `pypdf` for `pdfplumber` to fix PDF encoding issues
- Add ChromaDB or FAISS to persist embeddings across runs
- Implement recursive/semantic chunking
- Add conversation memory for follow-up questions
