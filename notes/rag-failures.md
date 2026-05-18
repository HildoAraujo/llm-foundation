## RAG Pipeline — Bugs & Failures Log

### Bug 1 — Deprecated OpenAI API pattern (`main.py`)
**Code:** `openai.api_key = os.getenv("OPENAI_API_KEY")`
**Error:** `TypeError: Could not resolve authentication method`
**Fix:** The new OpenAI SDK (v1.0+) dropped the global key pattern. Must instantiate a client: `openai_client = openai.OpenAI(api_key=...)`
**Lesson:** New SDK versions break old patterns silently at runtime, not at import time.

---

### Bug 2 — Wrong variable names in `.env`
**Code:** `OPENAI_API_KEYS = "..."` (trailing S)
**Error:** `openai.OpenAIError: Missing credentials`
**Fix:** Rename to `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`
**Lesson:** `os.getenv()` returns `None` silently when the key doesn't match — no warning, just a downstream crash.

---

### Bug 3 — Letter O instead of zero (`chunker.py`)
**Code:** `i = O`
**Error:** `NameError: name 'O' is not defined`
**Fix:** `i = 0`
**Lesson:** Classic typo. Also `i += size - overlap` was outside the while loop — would have caused an infinite loop if the NameError hadn't stopped it first.

---

### Bug 4 — Missing comma in list literals (`retriever.py`, `generator.py`)
**Code:** `"Dogs are animals" "Machine learning uses data"` (no comma)
**Error:** No error — Python silently concatenates adjacent string literals into one.
**Fix:** Add the comma between strings.
**Lesson:** One of Python's most dangerous silent bugs. The list had fewer items than expected with no warning.

---

### Bug 5 — Wrong import path depending on working directory (`retriever.py`)
**Code:** `from embedder import embed_text`
**Error:** `ModuleNotFoundError: No module named 'embedder'` (when imported from project root)
**Fix:** `from src.embedder import embed_text`
**Lesson:** Relative imports work when running a file directly but break when the file is imported as a module from a parent directory.

---

### Bug 6 — Syntax error in function signature (`loader.py`)
**Code:** `def clean_text(text: str): -> str:`
**Error:** `SyntaxError: invalid syntax`
**Fix:** `def clean_text(text: str) -> str:`
**Lesson:** The colon for the function body and the return type arrow got mixed up.

---

### Bug 7 — Wrong keyword argument for `embed_text` (`run_eval.py`)
**Code:** `embed_text(chunks, model_name=config["embedding_model"])`
**Error:** `TypeError: unexpected keyword argument 'model_name'`
**Fix:** `embed_text(chunks, model=config["embedding_model"])`
**Lesson:** Function signature mismatch — the parameter is `model`, not `model_name`.

---

### Bug 8 — Typo in variable name (`run_eval.py`)
**Code:** `embeddings=embedding` (missing the s)
**Error:** `NameError: name 'embedding' is not defined`
**Fix:** `embeddings=embeddings`
**Lesson:** Small typos in long variable names are easy to miss.

---

### Bug 9 — Wrong config keys in eval (`run_eval.py`)
**Code:** `config["chunk_size"]`, `config["chunk_overlap"]`
**Error:** `KeyError: 'chunk_size'`
**Fix:** `config["chunking"]["size"]`, `config["chunking"]["overlap"]` — the config uses a nested structure.
**Lesson:** Always check config structure before referencing keys — flat vs nested is a common mismatch.

---

### Bug 10 — `retrieve` returned a list, eval expected a dict (`retriever.py`)
**Code:** `retrieved_chunks = retrieval_result["chunks"]` — but `retrieve()` returned a plain list
**Error:** `TypeError: list indices must be integers, not str`
**Fix:** Updated `retrieve()` to return `{"chunks": [...], "chunk_ids": [...], "scores": [...]}` and updated `main.py` to unpack accordingly.
**Lesson:** When a function is reused across multiple callers, changing its return type requires updating all callers.

---

### Bug 11 — Invalid model name (`generator.py`)
**Code:** `model = "claude-3-5-sonnet-latest"`
**Error:** `anthropic.NotFoundError: model: claude-3-5-sonnet-latest`
**Fix:** `model = "claude-sonnet-4-6"`
**Lesson:** Model names are exact strings — aliases like `-latest` are not always supported. Always verify against the current API docs.

---

### What surprised me

**The PDF encoding** — `pypdf` read the text with spaces between every character (`I d e n t i f y i n g`). Despite that, retrieval hit 100% across all 5 eval questions. Cosine similarity finds semantic meaning even in noisy text.

**The generator was honest** — at `top_k=3` it said "I could not find the answer" instead of hallucinating. At `top_k=6` it found the BBVA finance example. One config value was the difference.
