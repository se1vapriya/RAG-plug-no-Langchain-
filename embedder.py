"""Embedding calls, batched and retried."""

import time
from typing import List

from openai import OpenAI

from config import config

_client = None


def client() -> OpenAI:
    """Lazy client so importing this module never requires credentials."""
    global _client
    if _client is None:
        config.validate()
        _client = OpenAI(api_key=config.google_api_key, base_url=config.api_base_url)
    return _client


def _create(inputs: List[str]) -> List[List[float]]:
    last_error = None
    for attempt in range(3):
        try:
            response = client().embeddings.create(
                input=inputs,
                model=config.embedding_model,
                dimensions=config.embedding_dimensions,
            )
            # The API may return items out of order; sort by index to be safe.
            return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        except Exception as e:  # transient rate limits are common here
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Embedding request failed after 3 attempts: {last_error}")


def embed_texts(texts: List[str], batch_size: int = None) -> List[List[float]]:
    """Embed many texts, batching to stay under per-request limits."""
    batch_size = batch_size or config.embedding_batch_size
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(_create(batch))
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return vectors


def embed_query(query: str) -> List[float]:
    return _create([query])[0]
