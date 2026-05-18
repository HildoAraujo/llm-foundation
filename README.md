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

Ran 5 configs against 14 questions (easy, implicit, multi-hop, specific numbers, 1 unanswerable negative test) on *Identifying and Scaling AI Use Cases*. The best config hit 79%, worst 71% — a realistic range that surfaces real retrieval gaps.

| Config | Chunk | Overlap | Top K | Hit Rate | Avg Score |
|---|---|---|---|---|---|
| baseline | 500 | 50 | 3 | 71% | 0.6226 |
| **wider_retrieval** | **500** | **50** | **6** | **79%** | **0.6226** |
| larger_chunks | 1000 | 100 | 3 | 71% | 0.6025 |
| smaller_chunks | 250 | 25 | 6 | 71% | 0.6314 |
| no_overlap | 500 | 0 | 3 | 71% | 0.6075 |

**Winner:** `wider_retrieval` (top_k=6, chunk=500). **Surprising finding:** smaller chunks (250) had the highest avg similarity score but same hit rate as baseline — more precise embeddings but not enough retrieved to cover all answers. Larger chunks scored worst across both metrics.

Run it yourself:

```bash
python -m evals.run_eval          # single config with answers
python evals/run_comparison.py    # 5-config sweep, retrieval only
```

Full results in `evals/results/comparison.md`.

## What I learned

**How RAG actually works**
Built every layer by hand — chunking, embedding, retrieval, generation. It's not magic: relevant text is injected into the prompt at runtime, and the LLM answers from that context only.

**Why bugs in AI pipelines are sneaky**
Most bugs here weren't crashes — a letter `O` instead of zero, a missing comma silently concatenating strings, a wrong variable name returning `None` quietly. AI pipelines fail silently more often than they crash loudly.

**PDF loader quality matters — but not equally**
`pypdf` produced mangled text (`I d e n t i f y i n g`). Both `pdfplumber` and `pymupdf` produce clean text. Switched to `pymupdf` as default. Despite the garbage text, embeddings were robust enough to hit on easy questions — but it hurt on implicit/multi-hop ones.

**`top_k` changes everything**
At `top_k=3` the system said "I can't find the answer." At `top_k=6` it found the BBVA finance example. One config value was the difference between a useful and a useless answer.

**A grounded system prompt prevents hallucination**
When context was missing, the generator said so instead of making something up. That's the system prompt doing its job — and it worked in practice.

**How to evaluate a RAG system**
Built an eval pipeline from scratch: keyword-based hit detection, cosine similarity scores per question, a markdown report. That's the same pattern used in production — measurable retrieval quality, not vibes.

**The AI engineering stack**
This project sits at the application layer — OpenAI and Anthropic APIs are the infrastructure, retrieval logic and prompt engineering are built on top. That's where most real-world AI products live.

## What's next

- Add proper retrieval metrics (Recall@k, MRR) beyond keyword hit rate
- LLM-as-judge for answer quality scoring
- Add ChromaDB or FAISS to persist embeddings across runs
- Implement recursive/semantic chunking
- Add conversation memory for follow-up questions
