## RAG Pipeline — Session Reflection

### What broke

**5 bugs across 4 files:**

1. **`openai.api_key = ...` (main.py)** — the new OpenAI SDK dropped the global API key pattern. Fix: instantiate `openai.OpenAI(api_key=...)` as a client, same pattern as Anthropic.

2. **`.env` variable names had a trailing `S`** (`OPENAI_API_KEYS`) — `os.getenv("OPENAI_API_KEY")` returned `None` silently. Fix: rename to `OPENAI_API_KEY`.

3. **`i = O` in chunker.py** — letter O instead of zero. The while loop never started. Also `i += size - overlap` was outside the loop, causing an infinite loop if it had run. Fix: `i = 0` and indent inside the while.

4. **Missing comma in list literals** — in both `retriever.py` and `generator.py`, two strings were silently concatenated into one. Python doesn't warn about this. Fix: add the comma.

5. **`from embedder import embed_text` in retriever.py** — works when run standalone from `src/`, breaks when imported from the project root. Fix: `from src.embedder import embed_text`.

---

### What surprised me

**The PDF encoding** — `pypdf` read the text with spaces between every character (`I d e n t i f y i n g`). Despite that, the embeddings and retrieval still worked. Cosine similarity is robust enough to find the right chunks even with mangled text. The BBVA finance example was retrieved correctly at `top_k=6`. Good surprise about how resilient embeddings are.

**The generator was honest** — at `top_k=3` it said "I could not find the answer in the provided context" rather than hallucinating. That's the system prompt working as intended.

---

### What I ended up with

- **PDF tested:** *Identifying and Scaling AI Use Cases* (AI adoption guide)
- **Question asked:** "What does the document say about research use cases in finance?"
- **Answer:** Correctly surfaced BBVA's Credit Analysis ProGPT — pulls unstructured data from annual reports, ESG assessments, and press sources to help credit risk analysts accelerate assessments
- **Did the answer look right?** Yes — grounded, specific, and limited to what was in the context
- **Repo:** No commits yet. `.env` is untracked — add it to `.gitignore` before pushing
