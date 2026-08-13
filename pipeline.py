"""The two operations every RAG project needs: ingest and ask.

    from pipeline import ingest, ask
    ingest()                       # index everything in DATA_DIR
    print(ask("What is the leave policy?")["answer"])
"""

from pathlib import Path
from typing import Dict, List

import chunker
import embedder
import llm
import loaders
import vectorstore
from config import config


def ingest(folder: Path = None, namespace: str = None, reset: bool = False) -> Dict:
    """Load -> chunk -> embed -> upsert. Returns a small summary."""
    folder = Path(folder) if folder else config.data_dir

    print(f"[1/4] Loading files from {folder}")
    records = loaders.load_folder(folder)
    sources = sorted({m.get("source") for _, m in records})
    print(f"      {len(records)} records from {len(sources)} file(s)")
    if not records:
        return {"files": 0, "chunks": 0, "vectors": 0}

    print("[2/4] Chunking")
    chunks = chunker.chunk_records(records)
    print(f"      {len(chunks)} chunks")

    if reset:
        print("      clearing existing namespace")
        vectorstore.clear(namespace)

    print("[3/4] Embedding")
    vectors = embedder.embed_texts([text for text, _ in chunks])

    print("[4/4] Upserting")
    count = vectorstore.upsert_records(chunks, vectors, namespace)

    print(f"Done. {count} vectors indexed from {len(sources)} file(s).")
    return {"files": len(sources), "chunks": len(chunks), "vectors": count, "sources": sources}


def retrieve(query: str, top_k: int = None, namespace: str = None) -> List[Dict]:
    return vectorstore.search(embedder.embed_query(query), top_k=top_k, namespace=namespace)


def ask(query: str, top_k: int = None, namespace: str = None) -> Dict:
    """Retrieve then generate. Returns {answer, hits, sources}."""
    query = (query or "").strip()
    if not query:
        return {"answer": "Please enter a question.", "hits": [], "sources": []}

    hits = retrieve(query, top_k=top_k, namespace=namespace)
    if not hits:
        return {
            "answer": "I couldn't find anything relevant in the knowledge base for that question.",
            "hits": [],
            "sources": [],
        }

    answer = llm.generate(query, llm.build_context(hits))
    sources = sorted({h["source"] for h in hits})
    return {"answer": answer, "hits": hits, "sources": sources}
