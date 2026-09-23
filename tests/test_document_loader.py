import pymupdf as fitz

from src.document_loader import _detect_printed_page_number, load_pdf_bytes


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


def test_detect_printed_page_number_finds_trailing_number():
    text = "Some report body text.\n© 2025 Example Commission 4"
    assert _detect_printed_page_number(text) == 4


def test_detect_printed_page_number_returns_none_without_trailing_number():
    text = "Some report body text that ends in a word."
    assert _detect_printed_page_number(text) is None


def test_detect_printed_page_number_ignores_table_row_ending_in_a_price():
    """Regression test: a table row whose last cell is a large price (e.g.
    a real catalog page ending "...| 4335") should not be mistaken for a
    printed page number -- unlike a genuine footer, prices routinely exceed
    the 999 bound and the row itself is short, which used to be enough to
    match the old, looser heuristic.
    """
    text = "Product | Size | Price\nD70 | 70 | 3090\nD75 | 75 | 4335"
    assert _detect_printed_page_number(text) is None


def test_detect_printed_page_number_finds_number_buried_in_short_table_row():
    """A genuine footer page number that landed as a stray trailing cell in
    an otherwise-short table row should still be detected.
    """
    text = "GH 1500 | 1500 | 2 | 747 | | | | | 13"
    assert _detect_printed_page_number(text) == 13


def test_detect_printed_page_number_ignores_long_last_line():
    long_line = " ".join(f"word{i}" for i in range(20)) + " 5"
    assert _detect_printed_page_number(long_line) is None


def test_load_pdf_bytes_keeps_two_column_sections_from_interleaving():
    """Regression test: a PDF whose content stream interleaves two
    side-by-side columns row-by-row should not merge both section headers
    together before either table's content (the bug that misattributed a
    product row to the wrong section in the real Taparia price list PDF).
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=400)
    rows = [
        (50, "SECTION-A-HEADER"),
        (350, "SECTION-B-HEADER"),
        (50, "A-ROW-1"),
        (350, "B-ROW-1"),
        (50, "A-ROW-2"),
        (350, "B-ROW-2"),
    ]
    y = 50
    for x, text in rows:
        page.insert_text((x, y), text, fontsize=9)
        y += 15
    pdf_bytes = doc.tobytes()
    doc.close()

    pages = load_pdf_bytes(pdf_bytes)
    text = pages[0].text

    # Both of section A's rows must appear before section B's header --
    # i.e. the columns stay grouped instead of interleaving by raw stream order.
    assert text.index("A-ROW-2") < text.index("SECTION-B-HEADER")
    assert text.index("SECTION-A-HEADER") < text.index("A-ROW-1") < text.index("A-ROW-2")
    assert text.index("SECTION-B-HEADER") < text.index("B-ROW-1") < text.index("B-ROW-2")
