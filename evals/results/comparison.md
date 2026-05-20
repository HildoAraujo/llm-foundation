# Config Comparison

Questions: 14 | Embedding: text-embedding-3-small | Loader: pymupdf

| Config | Strategy | Top K | Hit Rate | Avg Top Score |
|--------|----------|-------|----------|---------------|
| dense_k3 | dense | 3 | 71% | 0.6226 |
| dense_k6 | dense | 6 | 79% | 0.6226 |
| dense_rerank | dense_rerank | 3 | 71% | 0.7856 |
| bm25_only | bm25_only | 3 | 79% | 12.7169 |
| hybrid_k3 | hybrid | 3 | 79% | 0.0324 |
| hybrid_k6 | hybrid | 6 | 79% | 0.0324 |
| hybrid_rerank | hybrid_rerank | 3 | 71% | 0.7857 |

**Winner:** `bm25_only` — 79% hit rate, avg score 12.7169
**Worst:** `dense_k3` — 71% hit rate, avg score 0.6226