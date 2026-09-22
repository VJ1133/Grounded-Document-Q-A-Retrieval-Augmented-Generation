import pymupdf as fitz

from src.document_loader import load_pdf_bytes


def _make_pdf_bytes(pages_text: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_load_pdf_bytes_extracts_text_per_page():
    pdf_bytes = _make_pdf_bytes(["Hello insurance world", "Second page content"])
    pages = load_pdf_bytes(pdf_bytes)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Hello insurance world" in pages[0].text
    assert pages[1].page_number == 2
    assert "Second page content" in pages[1].text


def test_load_pdf_bytes_skips_blank_pages():
    pdf_bytes = _make_pdf_bytes(["Only content page"])
    doc = fitz.open("pdf", pdf_bytes)
    doc.insert_page(-1)  # trailing blank page
    pdf_bytes = doc.tobytes()
    doc.close()

    pages = load_pdf_bytes(pdf_bytes)

    assert len(pages) == 1
    assert pages[0].page_number == 1
