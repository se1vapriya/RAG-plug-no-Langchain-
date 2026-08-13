# RAG Template

A reusable retrieval-augmented-generation pipeline. Copy this folder, edit `.env`, drop documents in `data/`, run two commands.

Stack: Gemini (embeddings + generation via the OpenAI-compatible endpoint) and Pinecone.

## New project in 5 steps

```bash
cp -r rag-template my-new-project && cd my-new-project
python -m venv .venv && .venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                # then fill in the keys
# put your PDFs / .txt / .md into data/
python ingest.py
python ask.py "your question"
```

Web UI: `python -m uvicorn app:app --reload --port 8000` → http://127.0.0.1:8000

## Layout

| File | Role |
|---|---|
| `config.py` | Every setting, read from `.env`. **The only file you normally edit.** |
| `loaders.py` | File → text records with `source` / `page` metadata. Register new formats in `LOADERS`. |
| `chunker.py` | Overlapping character windows, snapped to word boundaries, metadata preserved. |
| `embedder.py` | Batched embedding calls with retry. |
| `vectorstore.py` | Pinecone upsert / query / clear. Content-hashed IDs. |
| `llm.py` | Answer generation with model fallback and numbered, citable context. |
| `pipeline.py` | `ingest()` and `ask()` — the whole thing in two functions. |
| `ingest.py` | CLI for indexing. |
| `ask.py` | CLI for querying (one-shot or interactive). |
| `app.py` | FastAPI: `/`, `/health`, `POST /api/chat`. |
| `static/index.html` | Minimal chat UI. Replace with your own frontend. |

## Per-project checklist

1. `PROJECT_NAME` in `.env`.
2. Create a Pinecone index with **dimension = `EMBEDDING_DIMENSIONS` (1536)**, **metric = cosine**. Set `PINECONE_INDEX_NAME`, or `PINECONE_INDEX_HOST` if the key belongs to a different Pinecone project.
3. `SYSTEM_PROMPT` — give the assistant its domain ("You are an HR policy assistant…"). This matters more than any other setting for answer quality.
4. `CHUNK_SIZE` / `CHUNK_OVERLAP` — 900/150 suits prose; drop to ~400/80 for dense tables or Q&A-style docs.
5. `TOP_K` — raise to 6–8 if answers miss context; `MIN_SCORE` ~0.3 filters junk matches.
6. `PINECONE_NAMESPACE` — use one namespace per project if you share a single index.

## Usage as a library

```python
from pipeline import ingest, ask

ingest("./docs", reset=True)

result = ask("What is the notice period?")
print(result["answer"])
for h in result["hits"]:
    print(h["source"], h["page"], round(h["score"], 3))
```

## Common operations

```bash
python ingest.py --path ./contracts --namespace legal   # separate corpus
python ingest.py --reset                                # rebuild from scratch
python ask.py -k 8 --show-sources "…"                   # widen retrieval, show scores
curl localhost:8000/health                              # vector count + config sanity check
```

## Adding a file format

```python
# loaders.py
def load_docx(path):
    import docx
    text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return [(text, {"source": path.name, "page": 1})] if text.strip() else []

LOADERS[".docx"] = load_docx
```

Then add `.docx` to `SUPPORTED_EXTENSIONS` in `.env`.

## Notes

- Vector IDs are a SHA-1 of source + page + chunk index + text, so re-ingesting unchanged files is idempotent. Renamed or deleted source files leave stale vectors behind — use `--reset` for those.
- `config.validate()` fails fast with a readable message rather than surfacing a provider 401.
- `llm.py` tries each model in `LLM_MODELS` in order and falls through on 429/5xx, which Gemini returns under load.
- Never commit `.env`; it is gitignored.
