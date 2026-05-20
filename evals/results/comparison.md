# Config Comparison

Questions: 14 | Embedding: text-embedding-3-small | Loader: pymupdf

| Config | Chunk | Overlap | Top K | Rerank | Hit Rate | Avg Top Score |
|--------|-------|---------|-------|--------|----------|---------------|
| baseline | 500 | 50 | 3 | no | 71% | 0.6226 |
| wider_retrieval | 500 | 50 | 6 | no | 79% | 0.6226 |
| rerank_k3_from_20 | 500 | 50 | 3 | yes | 71% | 0.7856 |
| rerank_k6_from_20 | 500 | 50 | 6 | yes | 79% | 0.7856 |
| rerank_k3_from_50 | 500 | 50 | 3 | yes | 71% | 0.7866 |

**Winner:** `rerank_k6_from_20` — 79% hit rate, avg score 0.7856
**Worst:** `baseline` — 71% hit rate, avg score 0.6226