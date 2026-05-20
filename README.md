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
│   ├── run_comparison.py       # 7-config sweep with per-question breakdown
│   └── results/comparison.md  # Latest comparison results
├── notes/
│   ├── rag-failures.md        # Bug log, loader comparison, eval analysis
│   └── findings.md            # Science journal: hypothesis → method → result per experiment
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

Five days of experiments across retrieval strategies, plus a manual per-question diagnosis that found 2 of 3 "ceiling" failures were eval design bugs — `must_contain` keywords that could never co-occur in a single chunk regardless of retrieval quality. After correcting those questions, the system runs at 86–93%. One genuine hard failure remains (Q11): the right chunk exists but the query's embedding doesn't reach it — a semantic gap between "use case doesn't perform as expected" and "feedback loops to iterate and optimize."

| Config | Strategy | Top K | Hit Rate | Avg Score |
|---|---|---|---|---|
| dense_k3 | dense | 3 | 86% | 0.6226 |
| **dense_k6** | **dense** | **6** | **93%** | **0.6226** |
| dense_rerank | dense_rerank | 3 | 86% | 0.7856 |
| **bm25_only** | **bm25_only** | **3** | **93%** | **—** |
| **hybrid_k3** | **hybrid** | **3** | **93%** | **—** |
| **hybrid_k6** | **hybrid** | **6** | **93%** | **—** |
| hybrid_rerank | hybrid_rerank | 3 | 86% | 0.7857 |

**Winner:** `dense_k6` / `bm25_only` / `hybrid` (tied at 93%, 13/14). The one remaining miss (Q11) is an embedding-level semantic gap — not fixable by retrieval tuning alone. Reranking hurts when layered on hybrid, dropping to 86%; the cross-encoder isn't calibrated for RRF-fused candidates.

```bash
python -m evals.run_eval          # single config with generated answers
python evals/run_comparison.py    # 7-config sweep across Days 3-5
```

Full results in `evals/results/comparison.md`.

## What I learned

**RAG is retrieval first, generation second.** The generator is only as good as what the retriever gives it. Most failures in this project were retrieval failures, not generation failures.

**Bugs in AI pipelines fail silently.** A letter `O` instead of zero, a missing comma that silently concatenated strings, a wrong variable name returning `None`. None of these crashed immediately — they just produced wrong results downstream.

**PDF loader quality matters.** `pypdf` produced `I d e n t i f y i n g` — spaces between every character. Both `pdfplumber` and `pymupdf` produced clean text. Embeddings are surprisingly robust to noise on easy questions, but loader quality shows up on implicit and multi-hop questions.

**`top_k` moves the needle more than chunk size.** Configs at `top_k=3` hit 86%; `top_k=6` hit 93%. Chunk size and overlap changes at the same `top_k` made no difference. Overlap removal hurt more than doubling chunk size — chunk boundaries are where answers live.

**Eval design bugs are indistinguishable from system failures.** What looked like a 71–79% ceiling across 5 days of retrieval experiments turned out to be 2 broken questions — `must_contain` keywords that could never co-occur in a single chunk regardless of any retrieval strategy. The per-question breakdown (not just aggregate hit rate) exposed this immediately. Always verify that your eval keywords actually appear in the document and can co-occur in a retrievable unit before concluding your system has a ceiling.

**Re-ranking improves confidence but not recall.** Adding a cross-encoder (BAAI/bge-reranker-base) lifted avg similarity scores from 0.62 to 0.79 — the model is returning more relevant chunks — but hit rate didn't move. The failing questions aren't being ranked wrong; their answer chunks aren't making it into the top-20 candidate pool at all. The ceiling is a recall problem, not a ranking problem. Next fix: semantic chunking or a stronger embedding model.

**BM25 is complementary until it isn't.** BM25 alone matched the best dense config (79%) — lexical matching recovered a question that embedding similarity missed. But hybrid didn't improve beyond 79%, meaning the two retrievers found the same 11 questions when combined. More importantly: hybrid+rerank dropped to 71%. The cross-encoder was not trained to rank fused RRF scores and actively made things worse. Stacking techniques compounds errors when each layer introduces its own failure mode.

**Negative tests are essential.** The system correctly refused to answer a question the document doesn't cover — across all 5 configs. That's the system prompt doing its job, and you can only verify it with a test designed to produce a miss.

## What's next

- LLM-as-judge for answer quality — keyword hit rate can't verify multi-hop reasoning (Q13)
- Fix Q11: investigate query expansion or a stronger embedding model for the semantic gap
- Add proper retrieval metrics (Recall@k, MRR) beyond keyword hit rate
- Add ChromaDB or FAISS to persist embeddings across runs
- Implement recursive/semantic chunking and compare against fixed chunking
