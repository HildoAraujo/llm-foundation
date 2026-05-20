# Findings

Science journal for the RAG pipeline project. Each entry: hypothesis, method, result, interpretation.

---

## Entry 1 — Does a cross-encoder re-ranker break the 71% hit-rate ceiling?

**Date:** 2026-05-20

**Hypothesis:** The 71% hit-rate ceiling from Day 3 is caused by bi-encoder compression loss. Bi-encoders (text-embedding-3-small) embed the query and each chunk independently, then compare vectors — fast but lossy for implicit and multi-hop questions. A cross-encoder looks at (query, chunk) jointly and should recover precision that cosine similarity loses.

**Method:** Added `src/reranker.py` using `sentence-transformers` CrossEncoder with `BAAI/bge-reranker-base`. Two-stage retrieval: embed → cosine similarity → top-20 candidates → cross-encoder rerank → top-k. Ran 5 configs: 2 Day 3 baselines (no rerank) + 3 reranker variants (k3_from_20, k6_from_20, k3_from_50). Eval: 14 questions, retrieval only (generate=False).

**Result:**

| Config | Top K | Rerank | Hit Rate | Avg Score |
|--------|-------|--------|----------|-----------|
| baseline | 3 | no | 71% | 0.6226 |
| wider_retrieval | 6 | no | 79% | 0.6226 |
| rerank_k3_from_20 | 3 | yes | 71% | 0.7856 |
| rerank_k6_from_20 | 6 | yes | 79% | 0.7856 |
| rerank_k3_from_50 | 3 | yes | 71% | 0.7866 |

**Interpretation:** The hypothesis was partially wrong. Reranking lifted avg confidence scores by ~26% (0.62 → 0.79), confirming the cross-encoder ranks more accurately. But hit rate didn't move — the same 10–11 questions pass and the same 3–4 fail across every config. This means the failing questions' answer chunks are not reaching the top-20 candidate pool from the bi-encoder at all. The bottleneck is recall at the embedding stage, not ranking precision. Re-ranking is the right production choice (better confidence, higher answer quality) but it doesn't fix the ceiling.

**What would fix it:** Semantic or recursive chunking (chunks that respect document structure rather than fixed character counts) could place answer content in retrievable units. A stronger embedding model (e.g. text-embedding-3-large, or a domain-tuned model) could improve recall on implicit and multi-hop questions. Worth testing next.

**Note on CPU latency:** On Apple Silicon CPU, each reranker pass over 20 candidates took ~1–2 seconds per question. Full sweep of 3 reranker configs × 14 questions ≈ 1–2 minutes. Production systems run cross-encoders on GPU or use API-based rerankers (Cohere Rerank). Not a bottleneck for offline evals but would be for real-time use.
