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

---

## Entry 2 — Does hybrid retrieval (BM25 + dense + RRF) break the 79% ceiling?

**Date:** 2026-05-20

**Hypothesis:** The 3 questions that all dense configs miss require lexical overlap that embedding similarity loses. BM25 ranks by term frequency weighted by inverse document frequency — it should recover questions containing exact proper nouns (BBVA, two-page) or specific numbers that embeddings blur into semantic neighborhoods.

**Method:** Added `src/bm25_retriever.py` (rank-bm25, simple lowercase + punctuation-strip tokenizer) and `src/hybrid_retriever.py` (RRF fusion with k=60). Tested 4 new configs: bm25_only (diagnostic), hybrid_k3, hybrid_k6, hybrid_rerank. All use chunk=500, overlap=50.

**Result:**

| Config | Strategy | Top K | Hit Rate |
|--------|----------|-------|----------|
| dense_k3 | dense | 3 | 71% |
| dense_k6 | dense | 6 | 79% |
| dense_rerank | dense_rerank | 3 | 71% |
| bm25_only | bm25_only | 3 | 79% |
| hybrid_k3 | hybrid | 3 | 79% |
| hybrid_k6 | hybrid | 6 | 79% |
| hybrid_rerank | hybrid_rerank | 3 | 71% |

**Interpretation:**

Three findings, each surprising:

1. **BM25 alone hit 79%.** It matched the best dense config without embeddings. This confirms BM25 is recovering at least one question that dense misses — likely a lexically specific question where the exact term appears in the document.

2. **Hybrid didn't improve beyond 79%.** Both retrievers find the same 11 questions. The 3 failing questions are hard for both BM25 and dense — they require understanding that neither term frequency nor cosine similarity can bridge with 500-char chunks. This is a chunking problem, not a retriever problem.

3. **hybrid_rerank dropped to 71%.** The cross-encoder actively hurt when layered on RRF-fused results. Most likely cause: BAAI/bge-reranker-base was not trained on documents structured like this, and RRF scores are on a different scale (~0.03) than the pairs it was trained to rank. It's re-ranking noise. Lesson: stacking retrieval techniques compounds errors unless each layer is calibrated to the previous layer's output distribution.

**What would fix it:** Semantic or recursive chunking (e.g. splitting at paragraph/section boundaries) would keep multi-sentence answers in the same chunk. The 3 failing questions likely have their answer spread across a chunk boundary. No retriever can fix that — it's a document representation problem.

**Next:** Implement recursive character text splitter (LangChain-style) or section-aware chunking. Compare against fixed chunking on the same 14 questions.

---

## Entry 3 — Manual diagnosis of the 3 universally-failing questions

**Date:** 2026-05-20

**Method:** Added per-question breakdown to `run_comparison.py`. Identified Q11, Q12, Q13 as 0/7 across all configs. Wrote diagnostic script to: (1) find each must_contain keyword in raw PDF text, (2) identify which chunk contains it, (3) check co-occurrence, (4) print actual top-6 retrieved chunks for the failing query.

**Findings:**

**Q11 — "pivot" — Embedding failure (chunk exists, query doesn't reach it).**
"pivot" appears exactly once: Chunk 45 — "05 Pivot and Scale The full team uses feedback loops to iterate and optimize based on GPT performance." The chunk is retrievable in principle. But the query "how should teams respond when an AI use case doesn't perform as expected" has no semantic overlap with "Pivot and Scale... GPT performance." Dense retrieval returns chunks 53, 7, 12, 8, 36, 49 — chunk 45 is not in the top 20. BM25 and hybrid also miss it (the word "pivot" doesn't appear in the query). Updated must_contain to ["feedback", "iterate"] — both in chunk 45. Q11 still fails 0/7: the retrieval failure is now confirmed independent of the keyword. This is a genuine embedding gap where the model doesn't connect "use case doesn't perform" to "iterate and optimize."

**Q12 — "leadership" + "champion" — Question design flaw (impossible by construction).**
"leadership" in Chunk 7. "champion" in Chunk 52. check_hit requires both keywords in a single retrieved chunk. No single chunk contains both — confirmed by exhaustive search. No retrieval strategy can pass this. Updated must_contain to ["leadership"] only. Q12 now passes 7/7.

**Q13 — "research" + "BBVA" — Multi-hop impossible under keyword-in-single-chunk check.**
"research" (as a primitive name) lives in Chunk 18 (the primitives list). "BBVA" lives in a separate BBVA use case chunk. They never co-occur. Answering this question correctly requires understanding that BBVA's Credit Analysis ProGPT *is* a Research use case — conceptual reasoning that keyword matching in a single chunk cannot verify. Updated must_contain to ["BBVA"] only. Q13 now passes 7/7. Full verification requires LLM-as-judge.

**Corrected eval results after question fixes:**

| Config | Hit Rate (old) | Hit Rate (corrected) |
|--------|---------------|----------------------|
| dense_k3 | 71% | 86% |
| dense_k6 | 79% | 93% |
| bm25_only | 79% | 93% |
| hybrid_k3 | 79% | 93% |
| hybrid_rerank | 71% | 86% |

**Interpretation:** Two of the three "ceiling" questions were eval bugs, not retrieval failures. The real system is performing at 86–93% on a corrected 14-question set. One genuine hard question remains (Q11) — an embedding-level semantic gap. The "79% ceiling" from Days 3–5 was partially self-inflicted by bad must_contain keywords.

**Lesson:** Eval design bugs are indistinguishable from system failures until you do the manual check. Always verify that must_contain keywords actually appear in the document and can co-occur in a single retrievable unit before treating a miss as a system failure.

**Next:** Semantic chunking may still help Q11 if the answer is near a section boundary — but the primary fix is likely a better embedding model or query expansion. Also: add LLM-as-judge as the verification method for multi-hop questions (Q13).
