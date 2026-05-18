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
pip install -e .
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

## Eval Results

Tested 4 configs across 5 questions on *Identifying and Scaling AI Use Cases* (PDF).

| Config | top_k | chunk_size | loader | Hit Rate | Avg Score |
|---|---|---|---|---|---|
| A | 3 | 500 | pymupdf | 80% | 0.6401 |
| **B** | **6** | **500** | **pymupdf** | **90%** | **0.6402** |
| C | 3 | 1000 | pymupdf | 80% | 0.6218 |
| D | 6 | 500 | pdfplumber | 80% | 0.6382 |

Tested on 10 questions (5 easy, 4 medium/hard, 1 unanswerable). The unanswerable question correctly returned "I could not find the answer" across all configs.

**Key findings:** `top_k` mattered more than chunk size or loader. Bigger chunks (C) scored worst — more text per chunk dilutes the embedding signal. pdfplumber (D) produced cleaner text but didn't outperform pymupdf on harder questions.

After expanding to 10 questions (including implicit answers, multi-hop, and one unanswerable question), hit rate dropped to **90%** — Q10 (legal contract review) correctly returned "I could not find the answer" because the document doesn't cover it. That's the eval working as intended.

Run it yourself:

```bash
python -m evals.run_eval        # single config
python evals/compare_configs.py # all 4 configs
```

Report saved to `evals/report.md`.

## What I learned

**How RAG actually works**
Built every layer by hand — chunking, embedding, retrieval, generation. It's not magic: relevant text is injected into the prompt at runtime, and the LLM answers from that context only.

**Why bugs in AI pipelines are sneaky**
Most bugs here weren't crashes — a letter `O` instead of zero, a missing comma silently concatenating strings, a wrong variable name returning `None` quietly. AI pipelines fail silently more often than they crash loudly.

**Embeddings are more robust than expected**
The PDF was read with spaces between every character (`I d e n t i f y i n g`) and retrieval still worked at 100% hit rate. Cosine similarity finds semantic meaning even in noisy text.

**`top_k` changes everything**
At `top_k=3` the system said "I can't find the answer." At `top_k=6` it found the BBVA finance example. One config value was the difference between a useful and a useless answer.

**A grounded system prompt prevents hallucination**
When context was missing, the generator said so instead of making something up. That's the system prompt doing its job — and it worked in practice.

**How to evaluate a RAG system**
Built an eval pipeline from scratch: keyword-based hit detection, cosine similarity scores per question, a markdown report. That's the same pattern used in production — measurable retrieval quality, not vibes.

**The AI engineering stack**
This project sits at the application layer — OpenAI and Anthropic APIs are the infrastructure, retrieval logic and prompt engineering are built on top. That's where most real-world AI products live.

## What's next

- Add harder eval questions — implicit answers, multi-hop, unanswerable — to stress-test beyond the easy baseline
- Add ChromaDB or FAISS to persist embeddings across runs
- Implement recursive/semantic chunking
- Add conversation memory for follow-up questions
