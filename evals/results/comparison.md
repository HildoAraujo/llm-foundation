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

## Per-Question Breakdown

| Question | Difficulty | dense_k3 | dense_k6 | dense_rerank | bm25_only | hybrid_k3 | hybrid_k6 | hybrid_rerank | Passes |
|---|---|---|---|---|---|---|---|---|---|
| Q1: What are the six AI use case primitives? | easy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q2: How does BBVA use AI for credit analysis? | easy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q3: What is the GPT Lab process at Estee Lauder? | easy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q4: What is automation as an AI use case? | easy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q5: How should companies gather and prioritize AI use cases | easy | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | 4/7 |
| Q6: How does AI help marketing teams work more efficiently  | medium | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q7: What role do subject matter experts play in building GP | medium | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q8: What makes an AI use case high impact but low effort? | medium | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q9: How many use cases were analyzed to identify the six pr | hard | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q10: What does the document say about cost savings from usin | negative | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q11: How should teams respond when an AI use case doesn't pe | implicit | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/7 |
| Q12: What does the document say about the role of leadership | implicit | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/7 |
| Q13: Which of the six AI use case primitives best describes  | multi-hop | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/7 |
| Q14: How long is the brief that business users create during | hard-number | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |