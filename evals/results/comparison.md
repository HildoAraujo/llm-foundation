# Config Comparison

Questions: 14 | Embedding: text-embedding-3-small | Loader: pymupdf

| Config | Chunking | Retrieval | Top K | Hit Rate | Avg Top Score |
|--------|----------|-----------|-------|----------|---------------|
| fixed_dense | fixed | dense | 3 | 86% | 0.6226 |
| fixed_dense_k6 | fixed | dense | 6 | 93% | 0.6226 |
| fixed_hybrid | fixed | hybrid | 3 | 93% | 0.0324 |
| recur_dense | recursive | dense | 3 | 71% | 0.6214 |
| recur_hybrid | recursive | hybrid | 3 | 93% | 0.0322 |
| sem_dense | semantic | dense | 3 | 93% | 0.6048 |
| sem_hybrid | semantic | hybrid | 3 | 93% | 0.0325 |

**Winner:** `fixed_dense_k6` — 93% hit rate, avg score 0.6226
**Worst:** `recur_dense` — 71% hit rate, avg score 0.6214

## Per-Question Breakdown

| Question | Difficulty | fixed_dense | fixed_dense_k6 | fixed_hybrid | recur_dense | recur_hybrid | sem_dense | sem_hybrid | Passes |
|---|---|---|---|---|---|---|---|---|---|
| Q1: What are the six AI use case primitives? | easy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q2: How does BBVA use AI for credit analysis? | easy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q3: What is the GPT Lab process at Estee Lauder? | easy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q4: What is automation as an AI use case? | easy | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q5: How should companies gather and prioritize AI use cases | easy | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | 5/7 |
| Q6: How does AI help marketing teams work more efficiently  | medium | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q7: What role do subject matter experts play in building GP | medium | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | 6/7 |
| Q8: What makes an AI use case high impact but low effort? | medium | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q9: How many use cases were analyzed to identify the six pr | hard | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q10: What does the document say about cost savings from usin | negative | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q11: How should teams respond when an AI use case doesn't pe | implicit | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 0/7 |
| Q12: What does the document say about the role of leadership | implicit | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | 6/7 |
| Q13: Which of the six AI use case primitives best describes  | multi-hop | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |
| Q14: How long is the brief that business users create during | hard-number | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7/7 |