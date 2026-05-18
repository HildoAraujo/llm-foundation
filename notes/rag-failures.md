## RAG Pipeline — Failures, Bugs & Eval Results

---

## Bugs Encountered

### Bug 1 — Deprecated OpenAI API pattern (`main.py`)
**Code:** `openai.api_key = os.getenv("OPENAI_API_KEY")`
**Error:** `TypeError: Could not resolve authentication method`
**Fix:** `openai_client = openai.OpenAI(api_key=...)`
**Lesson:** New SDK versions break old patterns silently at runtime, not at import time.

---

### Bug 2 — Wrong variable names in `.env`
**Code:** `OPENAI_API_KEYS = "..."` (trailing S)
**Error:** `openai.OpenAIError: Missing credentials`
**Fix:** Rename to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
**Lesson:** `os.getenv()` returns `None` silently when the key doesn't match — no warning, just a downstream crash.

---

### Bug 3 — Letter O instead of zero (`chunker.py`)
**Code:** `i = O`
**Error:** `NameError: name 'O' is not defined`
**Fix:** `i = 0`. Also `i += size - overlap` was outside the while loop — would have caused an infinite loop.
**Lesson:** AI pipelines fail silently more often than they crash loudly. This one at least crashed.

---

### Bug 4 — Missing comma in list literals (`retriever.py`, `generator.py`)
**Code:** `"Dogs are animals" "Machine learning uses data"` (no comma)
**Error:** None — Python silently concatenates adjacent strings into one.
**Fix:** Add the comma.
**Lesson:** One of Python's most dangerous silent bugs. The list had fewer items than expected with zero warning.

---

### Bug 5 — Import path breaks when called from project root (`retriever.py`)
**Code:** `from embedder import embed_text`
**Error:** `ModuleNotFoundError: No module named 'embedder'`
**Fix:** `from src.embedder import embed_text`
**Lesson:** Relative imports work when running a file directly but break when the file is imported as a module from a parent directory. This was the hardest bug to track down — ate the most time.

---

### Bug 6 — Syntax error in function signature (`loader.py`)
**Code:** `def clean_text(text: str): -> str:`
**Error:** `SyntaxError`
**Fix:** `def clean_text(text: str) -> str:`
**Lesson:** The colon for the function body and the return type arrow got mixed up.

---

### Bug 7 — Wrong kwarg name for `embed_text` (`run_eval.py`)
**Code:** `embed_text(chunks, model_name=config["embedding_model"])`
**Error:** `TypeError: unexpected keyword argument 'model_name'`
**Fix:** `embed_text(chunks, model=config["embedding_model"])`

---

### Bug 8 — Typo in variable name (`run_eval.py`)
**Code:** `embeddings=embedding`
**Error:** `NameError: name 'embedding' is not defined`
**Fix:** `embeddings=embeddings`

---

### Bug 9 — Wrong config keys in eval (`run_eval.py`)
**Code:** `config["chunk_size"]`, `config["chunk_overlap"]`
**Error:** `KeyError: 'chunk_size'`
**Fix:** `config["chunking"]["size"]`, `config["chunking"]["overlap"]`
**Lesson:** Flat vs nested config structure mismatch — always check shape before referencing keys.

---

### Bug 10 — `retrieve()` return type mismatch (`retriever.py`)
**Code:** `retrieval_result["chunks"]` — but `retrieve()` returned a plain list
**Error:** `TypeError: list indices must be integers, not str`
**Fix:** Updated `retrieve()` to return `{"chunks": [...], "chunk_ids": [...], "scores": [...]}` and updated all callers.
**Lesson:** Changing a function's return type requires updating every caller — `main.py` broke silently until tested.

---

### Bug 11 — Invalid model name (`generator.py`)
**Code:** `model = "claude-3-5-sonnet-latest"`
**Error:** `anthropic.NotFoundError: model: claude-3-5-sonnet-latest`
**Fix:** `model = "claude-sonnet-4-6"`
**Lesson:** Model name aliases like `-latest` are not always supported. Verify against current API docs.

---

## PDF Loader Comparison

Tested all three loaders on first 300 characters of *Identifying and Scaling AI Use Cases*:

- **pypdf** — `I d e n t i f y i n g a n d s c a l i n g A I` (garbage, spaces between every character)
- **pdfplumber** — `Identifying and scaling AI use cases How early adopters...` (clean)
- **pymupdf** — `Identifying and scaling AI use cases How early adopters...` (clean)

**Winner: pymupdf** (set as default). Both pdfplumber and pymupdf produce clean text. pypdf is unusable for this PDF type. The mangled text degraded retrieval on harder/implicit questions even though easy questions still hit — a hidden failure mode.

---

## Eval Results — 5-Config Sweep (14 questions)

Tested on 14 questions: 5 easy, 3 medium, 2 implicit, 2 multi-hop, 1 specific number, 1 negative test.
Hit rate on negative test = correctly refusing to answer (system said "I could not find the answer").

| Config | Chunk | Overlap | Top K | Hit Rate | Avg Score |
|---|---|---|---|---|---|
| baseline | 500 | 50 | 3 | 71% | 0.6226 |
| **wider_retrieval** | **500** | **50** | **6** | **79%** | **0.6226** |
| larger_chunks | 1000 | 100 | 3 | 71% | 0.6025 |
| smaller_chunks | 250 | 25 | 6 | 71% | 0.6314 |
| no_overlap | 500 | 0 | 3 | 71% | 0.6075 |

---

## What the numbers revealed

**`top_k=6` was the only config that beat baseline.**
Every config at `top_k=3` landed at 71%. Wider retrieval (top_k=6) was the only lever that moved the needle — from 71% to 79%. Chunk size and overlap changes made no difference at the same top_k.

**Smaller chunks had the best avg similarity score but same hit rate.**
`smaller_chunks` (chunk=250) had the highest avg score (0.6314) — tighter, more focused embeddings. But at top_k=6, it still only hit 71%. The precision was there; the recall ceiling was elsewhere.

**Larger chunks scored worst across all metrics.**
`larger_chunks` (chunk=1000) had the lowest avg score (0.6025) and same 71% hit rate. More text per chunk = embedding that's about everything = precise at nothing.

**Removing overlap hurt.**
`no_overlap` scored second worst (0.6075). Overlap exists because answers often sit at chunk boundaries — without it, you silently lose context at every seam.

**Hardest question categories:**
- **Multi-hop** (Q13 — "Which primitive describes BBVA's ProGPT?"): missed consistently — requires combining the primitives list AND the BBVA description in retrieved chunks simultaneously
- **Implicit** (Q12 — leadership/champion): hit rate inconsistent — "champion" appears infrequently and in specific chunks that weren't always retrieved
- **Specific numbers** (Q14 — "two-page brief"): hit on most configs — the term was in the Estée Lauder chunk that retrieves reliably

---

## What surprised me

**Easy questions masked real retrieval problems.**
A 5-question eval at 100% looks great. A 14-question eval with implicit and multi-hop questions at 71-79% shows the actual ceiling. The system doesn't understand — it pattern-matches. Multi-hop questions expose that directly.

**Overlap matters more than chunk size.**
Removing overlap (`no_overlap`) hurt more than doubling chunk size. Chunk boundaries are where answers live — and overlap is what ensures they're captured.

**The negative test passed every time.**
Q10 (legal contract review) was correctly refused across all 5 configs. The system prompt held even with harder questions around it.
