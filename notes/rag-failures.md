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

## Eval Results — Config Comparison

Tested across 5 questions on *Identifying and Scaling AI Use Cases* (PDF).
Hit = retrieved chunk contained all required keywords for that question.

| Config | top_k | chunk_size | loader | Hits | Hit Rate | Avg Score |
|---|---|---|---|---|---|---|
| A | 3 | 500 | pymupdf | 4/5 | 80% | 0.6637 |
| B | 6 | 500 | pymupdf | 5/5 | **100%** | 0.6637 |
| C | 3 | 1000 | pymupdf | 4/5 | 80% | 0.6359 |
| D | 6 | 500 | pdfplumber | 4/5 | 80% | **0.6922** |

---

## What the numbers revealed

**`top_k` mattered more than chunk size.**
Config A (top_k=3, chunk=500) and Config C (top_k=3, chunk=1000) both hit 80%. Doubling the chunk size didn't help — it just made each chunk noisier and lowered the average similarity score (0.6637 → 0.6359). The retrieval ceiling was hit by not fetching enough chunks, not by chunk size.

**Bigger chunks hurt score quality.**
Config C had the lowest average similarity score (0.6359) despite larger chunks. More text per chunk dilutes the embedding signal — the chunk becomes about many things at once instead of one specific thing.

**pdfplumber had higher similarity scores but same hit rate as top_k=3.**
Config D (pdfplumber) produced the highest average score (0.6922) — cleaner text means tighter embeddings. But it still missed 1 question. The combination of better loader + more retrieved chunks (top_k=6) would likely hit 100%.

**The one miss across configs A, C, D was the same question.**
Q3 — *"What is the GPT Lab process at Estée Lauder?"* — was the hardest to retrieve. The keyword `Estée Lauder` with the accent caused matching issues across some chunks. A preprocessing step to normalize accented characters would fix this.

---

## What surprised me

**Cosine similarity is robust to garbage text.**
`pypdf` produced `I d e n t i f y i n g` — spaces between every character. Config B still hit 100%. The embeddings found semantic meaning even in mangled text. That's genuinely unexpected.

**The generator was honest when context was missing.**
At top_k=3 it said "I could not find the answer" instead of making something up. The system prompt held. Good surprise.

**Larger chunks hurt, not helped.**
Intuition says more context per chunk = better retrieval. The numbers said the opposite — smaller, focused chunks produced better similarity scores and equal or better hit rates.
