# Config Comparison

Questions: 14 | Embedding: text-embedding-3-small | Loader: pymupdf

| Config | Chunk | Overlap | Top K | Hit Rate | Avg Top Score |
|--------|-------|---------|-------|----------|---------------|
| baseline | 500 | 50 | 3 | 71% | 0.6226 |
| wider_retrieval | 500 | 50 | 6 | 79% | 0.6226 |
| larger_chunks | 1000 | 100 | 3 | 71% | 0.6025 |
| smaller_chunks | 250 | 25 | 6 | 71% | 0.6314 |
| no_overlap | 500 | 0 | 3 | 71% | 0.6075 |

**Winner:** `wider_retrieval` — 79% hit rate, avg score 0.6226
**Worst:** `larger_chunks` — 71% hit rate, avg score 0.6025