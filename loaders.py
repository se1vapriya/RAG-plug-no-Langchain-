"""Turn files on disk into (text, metadata) records.

A record is one page for PDFs and one whole file for text formats. Metadata
travels with the text all the way into the vector store, so answers can cite
the source file and page.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import config

Record = Tuple[str, Dict]


def load_pdf(path: Path) -> List[Record]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    records: List[Record] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            records.append((text, {"source": path.name, "page": page_num}))
    return records


def load_text(path: Path) -> List[Record]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(text, {"source": path.name, "page": 1})] if text else []


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
}


def load_file(path: Path) -> List[Record]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader registered for '{path.suffix}' ({path.name})")
    return loader(path)


def iter_files(folder: Path, extensions: Iterable[str] = None) -> List[Path]:
    """Every supported file under `folder`, recursively, in stable order."""
    exts = {e.lower() for e in (extensions or config.supported_extensions)}
    if not folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {folder}")
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def load_folder(folder: Path = None, extensions: Iterable[str] = None) -> List[Record]:
    folder = Path(folder) if folder else config.data_dir
    records: List[Record] = []
    for path in iter_files(folder, extensions):
        records.extend(load_file(path))
    return records
