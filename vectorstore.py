"""Pinecone upsert / query, with deterministic IDs and metadata passthrough."""

import hashlib
from typing import Dict, List, Tuple

from pinecone import Pinecone

from config import config

Record = Tuple[str, Dict]

_index = None


def index():
    """Lazy index handle.

    Connecting by host skips the name lookup, which requires the API key to
    belong to the same Pinecone project as the index.
    """
    global _index
    if _index is None:
        config.validate()
        pc = Pinecone(api_key=config.pinecone_api_key)
        _index = (
            pc.Index(host=config.pinecone_index_host)
            if config.pinecone_index_host
            else pc.Index(config.pinecone_index_name)
        )
    return _index


def _vector_id(text: str, meta: Dict) -> str:
    """Content-hashed ID so re-ingesting the same file updates instead of
    duplicating, and edited content lands on a new vector."""
    key = f"{meta.get('source','')}|{meta.get('page','')}|{meta.get('chunk_index','')}|{text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def upsert_records(records: List[Record], embeddings: List[List[float]], namespace: str = None):
    namespace = config.namespace if namespace is None else namespace
    vectors = [
        {
            "id": _vector_id(text, meta),
            "values": vector,
            "metadata": {**meta, "text": text},
        }
        for (text, meta), vector in zip(records, embeddings)
    ]

    batch = config.upsert_batch_size
    for i in range(0, len(vectors), batch):
        index().upsert(vectors=vectors[i : i + batch], namespace=namespace)
        print(f"  upserted {min(i + batch, len(vectors))}/{len(vectors)}")
    return len(vectors)


def search(query_vector: List[float], top_k: int = None, namespace: str = None) -> List[Dict]:
    """Return [{text, score, source, page}, ...] above config.min_score."""
    results = index().query(
        vector=query_vector,
        top_k=top_k or config.top_k,
        include_metadata=True,
        namespace=config.namespace if namespace is None else namespace,
    )
    hits = []
    for match in results.matches:
        if match.score < config.min_score:
            continue
        meta = match.metadata or {}
        hits.append(
            {
                "text": meta.get("text", ""),
                "score": match.score,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
            }
        )
    return hits


def stats() -> Dict:
    return index().describe_index_stats()


def clear(namespace: str = None):
    """Delete every vector in the namespace. Useful when re-ingesting a
    corpus whose files were renamed or removed."""
    index().delete(
        delete_all=True, namespace=config.namespace if namespace is None else namespace
    )
