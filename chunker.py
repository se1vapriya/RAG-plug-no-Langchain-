"""Split loaded records into overlapping chunks, preserving metadata.

Chunking happens per record (per page / per file) rather than over one big
concatenated string, so a chunk never straddles two different sources and its
`source`/`page` metadata stays accurate.
"""

from typing import Dict, List, Tuple

from config import config

Record = Tuple[str, Dict]


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """Fixed-size character windows with overlap, snapped to whitespace.

    Snapping the cut to the nearest space avoids slicing a word in half, which
    otherwise shows up as garbled tokens in retrieved context.
    """
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = " ".join(text.split())
    length = len(text)
    if length == 0:
        return []
    if length <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            space = text.rfind(" ", start + chunk_overlap, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_records(
    records: List[Record], chunk_size: int = None, chunk_overlap: int = None
) -> List[Record]:
    """Chunk every record, carrying metadata plus a per-source chunk index."""
    out: List[Record] = []
    for text, meta in records:
        for i, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            out.append((chunk, {**meta, "chunk_index": i}))
    return out
