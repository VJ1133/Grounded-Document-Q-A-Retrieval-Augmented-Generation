"""Splits page text into overlapping word-based chunks for embedding."""

from dataclasses import dataclass

from src.document_loader import PageText


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int  # position within the document, for stable IDs


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    """Fixed-size word chunking with overlap, one page's text at a time.

    Chunking per page (rather than concatenating the whole document first)
    keeps every chunk attributable to a single page number for citations.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    chunk_index = 0
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(text=chunk_text, page_number=page.page_number, chunk_index=chunk_index)
            )
            chunk_index += 1
            if end == len(words):
                break
            start = end - overlap
    return chunks
