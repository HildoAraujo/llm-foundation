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
│   ├── chunker.py      # Fixed, recursive, and semantic chunking (dispatcher)
│   ├── semantic_chunker.py  # Sentence-embedding topic-shift chunker
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

Six days of experiments, each eliminating one hypothesis about where the ceiling comes from. Act 1 found an apparent 71–79% ceiling that chunk tuning couldn't move. Act 2 (reranking) lifted confidence scores but not hit rate — a recall problem, not a precision problem. Act 3 (BM25 hybrid) matched the best dense config but didn't improve further — both retrievers found the same questions. Act 4 (chunking): semantic chunking at top_k=3 matched what fixed chunking needed top_k=6 to achieve — better chunk quality is a substitute for wider retrieval. But Q11 failed across all 7 configs. Manual inspection confirmed 0 of the 2 remaining failures span chunk boundaries. The 93% ceiling is real and driven by a single embedding gap: the query "use case doesn't perform as expected" and the answer chunk "feedback loops to iterate and optimize" don't share a vector neighborhood in `text-embedding-3-small`.

| Config | Chunking | Retrieval | Top K | Hit Rate |
|---|---|---|---|---|
| fixed_dense | fixed | dense | 3 | 86% |
| fixed_dense_k6 | fixed | dense | 6 | 93% |
| fixed_hybrid | fixed | hybrid | 3 | 93% |
| recur_dense | recursive | dense | 3 | 71% |
| recur_hybrid | recursive | hybrid | 3 | 93% |
| **sem_dense** | **semantic** | **dense** | **3** | **93%** |
| sem_hybrid | semantic | hybrid | 3 | 93% |

**Winner:** `sem_dense` — semantic chunking + dense retrieval at top_k=3 = 93%. Same performance as wider retrieval, fewer API calls. Recursive chunking hurt at top_k=3 (content fragmentation) but recovered with hybrid. Q11 fails every config: confirmed embedding gap, not an architecture problem.

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

**Re-ranking improves confidence but not recall.** Adding a cross-encoder (BAAI/bge-reranker-base) lifted avg similarity scores from 0.62 to 0.79 — the model is returning more relevant chunks — but hit rate didn't move. The failing questions aren't being ranked wrong; their answer chunks aren't making it into the top-20 candidate pool at all. The ceiling is a recall problem, not a ranking problem.

**BM25 is complementary until it isn't.** BM25 alone matched the best dense config — lexical matching recovered a question that embedding similarity missed. But hybrid didn't improve beyond that, meaning both retrievers found the same questions when combined. Hybrid+rerank dropped hit rate: the cross-encoder isn't calibrated for RRF-fused candidates and actively made things worse. Stacking techniques compounds errors when each layer introduces its own failure mode.

**Semantic chunking is a substitute for wider retrieval.** `sem_dense` at top_k=3 matched `fixed_dense` at top_k=6 — more coherent chunks need fewer retrieved candidates to cover the answer. Recursive chunking *hurt* at top_k=3 (fragmented answers across boundaries) but recovered with hybrid. The choice of chunking strategy interacts with retrieval width: better chunks reduce how wide you need to cast the net.

**Negative tests are essential.** The system correctly refused to answer a question the document doesn't cover — across all 5 configs. That's the system prompt doing its job, and you can only verify it with a test designed to produce a miss.

## What's next

- Fix Q11: query expansion or a stronger embedding model for the semantic gap between "doesn't perform as expected" and "feedback loops to iterate"
- LLM-as-judge for answer quality — keyword hit rate can't verify multi-hop reasoning (Q13)
- Add proper retrieval metrics (Recall@k, MRR) beyond keyword hit rate
- Add ChromaDB or FAISS to persist embeddings across runs
