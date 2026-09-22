"""Extracts page-level text from a PDF using PyMuPDF."""

from dataclasses import dataclass

import pymupdf as fitz


@dataclass
class PageText:
    page_number: int  # 1-indexed, matches what a human would cite
    text: str


def load_pdf_pages(pdf_path: str) -> list[PageText]:
    """Return one PageText per page of the PDF, skipping blank pages."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(page_number=index + 1, text=text))
    return pages


def load_pdf_bytes(pdf_bytes: bytes) -> list[PageText]:
    """Same as load_pdf_pages but reads from in-memory bytes (Streamlit upload)."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(page_number=index + 1, text=text))
    return pages
