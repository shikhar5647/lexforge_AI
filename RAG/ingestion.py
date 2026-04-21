
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import fitz  # PyMuPDF
from docx import Document


@dataclass
class Chunk:
    text: str
    doc_id: str
    source_path: str
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    extra: dict = field(default_factory=dict)


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_txt(path: Path) -> str:
    return normalize_whitespace(path.read_text(encoding="utf-8", errors="replace"))


def load_pdf(path: Path) -> tuple[str, list[tuple[int, str]]]:
    """Returns full text and per-page segments for page metadata."""
    pages: list[tuple[int, str]] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            raw = page.get_text("text") or ""
            pages.append((i + 1, normalize_whitespace(raw)))
    full = normalize_whitespace("\n\n".join(t for _, t in pages if t))
    return full, pages


def load_docx(path: Path) -> str:
    d = Document(str(path))
    parts: list[str] = []
    for p in d.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return normalize_whitespace("\n\n".join(parts))


def load_document(path: Path) -> tuple[str, list[tuple[int, str]] | None]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".docx",):
        return load_docx(path), None
    if suffix in (".txt", ".md", ".text"):
        return load_txt(path), None
    raise ValueError(f"Unsupported format: {path} (use .pdf, .docx, .txt, .md)")


def recursive_split(
    text: str,
    chunk_size: int = 1400,
    chunk_overlap: int = 200,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """Split long legal text; prefers paragraph and line boundaries."""
    if separators is None:
        separators = ("\n\n## ", "\n\n# ", "\n\n", "\n", ". ", " ", "")
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    rest = text

    while rest:
        if len(rest) <= chunk_size:
            piece = rest.strip()
            if piece:
                chunks.append(piece)
            break

        window = rest[:chunk_size]
        cut = chunk_size
        for sep in separators:
            if not sep:
                cut = chunk_size
                break
            idx = window.rfind(sep)
            if idx >= chunk_size // 2:
                cut = idx + len(sep)
                break

        piece = rest[:cut].strip()
        if piece:
            chunks.append(piece)
        advance = cut - chunk_overlap
        if advance <= 0:
            advance = cut
        rest = rest[advance:].strip()

    return chunks


def assign_page_range(
    chunk_text: str,
    pages: list[tuple[int, str]] | None,
) -> tuple[int | None, int | None]:
    if not pages:
        return None, None
    starts: list[int] = []
    for pn, body in pages:
        if not body:
            continue
        if body[: min(80, len(body))] in chunk_text or chunk_text[:80] in body:
            starts.append(pn)
        elif any(s in chunk_text for s in body.split()[:5] if len(s) > 12):
            starts.append(pn)
    if not starts:
        return None, None
    return min(starts), max(starts)


def ingest_paths(
    paths: Iterable[Path | str],
    chunk_size: int = 1400,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    out: list[Chunk] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            continue
        doc_id = path.stem
        full, per_page = load_document(path)
        if not full:
            continue
        pieces = recursive_split(full, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            ps, pe = assign_page_range(piece, per_page)
            out.append(
                Chunk(
                    text=piece,
                    doc_id=doc_id,
                    source_path=str(path),
                    chunk_index=i,
                    page_start=ps,
                    page_end=pe,
                )
            )
    return out


def iter_supported_files(folder: Path) -> Iterator[Path]:
    folder = folder.resolve()
    if not folder.is_dir():
        yield from ()
        return
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt", ".md", ".text"}:
            yield p
