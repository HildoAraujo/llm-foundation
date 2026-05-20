# LLM Foundations — RAG Pipeline

A retrieval-augmented generation (RAG) pipeline built from scratch to understand how LLMs work in production. Every layer is hand-rolled — no LangChain, no abstractions.

```
PDF → Chunk → Embed → Retrieve → Generate → Answer
```

## Stack

| Layer | What it does | Model / Tool |
|---|---|---|
| Loader | Reads PDF, cleans text | `pymupdf` (default), `pdfplumber`, `pypdf` |
| Chunker | Splits text into overlapping fixed-size chunks | Custom |
| Embedder | Converts chunks to vectors | `text-embedding-3-small` (OpenAI) |
| Retriever | Finds top-k most similar chunks via cosine similarity | `scikit-learn` |
| Generator | Answers question from retrieved context only | `claude-sonnet-4-6` (Anthropic) |

## Project Structure

```
├── src/
│   ├── loader.py       # PDF loaders (3 options, switchable via config)
│   ├── chunker.py      # Fixed-size chunking with overlap
│   ├── embedder.py     # OpenAI embeddings
│   ├── retriever.py    # Cosine similarity retrieval
│   └── generator.py   # Claude answer generation
├── evals/
│   ├── questions.json          # 14-question eval set (easy → multi-hop → negative)
│   ├── run_eval.py             # Single-config eval with generated answers
│   ├── run_comparison.py       # 5-config sweep, retrieval only
│   └── results/comparison.md  # Latest comparison results
├── notes/
│   └── rag-failures.md        # Bug log, loader comparison, eval analysis
├── config.yaml
└── main.py
```

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

Add a PDF to `data/` and update `config.yaml`:

```yaml
pdf_path: "data/your-document.pdf"
loader: "pymupdf"
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

Ran 5 configs against 14 questions — easy, implicit, multi-hop, specific numbers, and 1 unanswerable negative test — on *Identifying and Scaling AI Use Cases*. Best config: 79%, worst: 71%. A realistic range that surfaces real retrieval gaps.

| Config | Chunk | Overlap | Top K | Hit Rate | Avg Score |
|---|---|---|---|---|---|
| baseline | 500 | 50 | 3 | 71% | 0.6226 |
| **wider_retrieval** | **500** | **50** | **6** | **79%** | **0.6226** |
| larger_chunks | 1000 | 100 | 3 | 71% | 0.6025 |
| smaller_chunks | 250 | 25 | 6 | 71% | 0.6314 |
| no_overlap | 500 | 0 | 3 | 71% | 0.6075 |

**Winner:** `wider_retrieval` (top_k=6, chunk=500). Surprising finding: smaller chunks (250) had the highest avg similarity score but the same hit rate — more precise embeddings, but not enough retrieved to cover multi-hop answers. Larger chunks scored worst on both metrics.

```bash
python -m evals.run_eval          # single config with generated answers
python evals/run_comparison.py    # 5-config sweep, retrieval only
```

Full results in `evals/results/comparison.md`.

## What I learned

**RAG is retrieval first, generation second.** The generator is only as good as what the retriever gives it. Most failures in this project were retrieval failures, not generation failures.

**Bugs in AI pipelines fail silently.** A letter `O` instead of zero, a missing comma that silently concatenated strings, a wrong variable name returning `None`. None of these crashed immediately — they just produced wrong results downstream.

**PDF loader quality matters.** `pypdf` produced `I d e n t i f y i n g` — spaces between every character. Both `pdfplumber` and `pymupdf` produced clean text. Embeddings are surprisingly robust to noise on easy questions, but loader quality shows up on implicit and multi-hop questions.

**`top_k` moved the needle more than chunk size.** Every config at `top_k=3` hit 71%. `top_k=6` hit 79%. Chunk size changes at the same `top_k` made no difference to hit rate. Overlap removal hurt more than doubling chunk size — chunk boundaries are where answers live.

**Five configs all hit a ceiling at 71%.** Every chunk size and overlap variation stalled at the same floor. Only `top_k=6` broke through to 79% — meaning retrieval width mattered, but configuration tuning alone couldn't push further. That's a signal the bottleneck is below the retrieval layer, likely at the embedding level: cosine similarity can't distinguish closely related concepts in a dense document, and no chunk size fixes that. The 71% ceiling is the next problem to solve.

**Negative tests are essential.** The system correctly refused to answer a question the document doesn't cover — across all 5 configs. That's the system prompt doing its job, and you can only verify it with a test designed to produce a miss.

## What's next

- Add proper retrieval metrics (Recall@k, MRR) beyond keyword hit rate
- LLM-as-judge for answer quality scoring
- Add ChromaDB or FAISS to persist embeddings across runs
- Implement recursive/semantic chunking
- Add conversation memory for follow-up questions
