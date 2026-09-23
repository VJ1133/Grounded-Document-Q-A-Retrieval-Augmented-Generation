"""Extracts page-level text from a PDF using PyMuPDF.

Plain `page.get_text("text")` reads content in PDF content-stream order,
which for multi-column layouts (side-by-side tables, catalog pages) often
does NOT match true left-to-right, top-to-bottom reading order. A row from
a right-hand table can end up textually adjacent to a left-hand table's
section header, misattributing it. `_extract_page_text` fixes this for the
common case (two side-by-side columns) by:

  1. Detecting real tables with PyMuPDF's find_tables() and formatting their
     rows explicitly (so cell alignment survives instead of flattening into
     an ambiguous word soup).
  2. Bucketing every text block and table by which half of the page it
     starts in (left vs. right), then emitting all of the left column's
     content top-to-bottom before the right column's -- instead of raw
     stream order, which can interleave the two.

Single-column pages are unaffected: everything buckets "left" and the
result is the same top-to-bottom order as before.
"""

import contextlib
import io
import re
from dataclasses import dataclass

import pymupdf as fitz

# Matches a standalone number (1-4 digits) as the very last token of a line.
_TRAILING_PAGE_NUMBER = re.compile(r"(?<!\S)(\d{1,4})\s*$")

# A genuine printed page number is small and sits on a short line (a footer,
# not a data row). On table-heavy pages the LAST line is often a table row
# whose last cell happens to be a price/quantity -- e.g. "...| 4335" -- which
# looks identical to a page number by pattern alone. These two bounds are a
# pragmatic filter against that: real footers are short, and real page
# counts for this kind of document are well under four digits, while catalog
# prices routinely run into the thousands.
_MAX_PLAUSIBLE_PRINTED_PAGE = 999
_MAX_FOOTER_LINE_WORDS = 12


def _detect_printed_page_number(text: str) -> int | None:
    """Best-effort guess at the page number printed on the page itself.

    Many reports have front matter (cover page, table of contents, executive
    summary) before their own page "1", so the physical position of a page in
    the PDF often does not match the number printed in its footer. This
    heuristic looks at the LAST non-empty line of the page's extracted text
    -- where a footer usually lands -- and returns its trailing number only
    if that line is short and the number is a plausible page count. It can
    still be wrong on some layouts, so it's a display aid, not a guarantee --
    always fall back to the physical page_number when this returns None.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    # Table rows are formatted with literal " | " separators (see
    # _format_table); strip those out before counting words so a short,
    # mostly-empty table row isn't miscounted as a long line.
    word_count = len([w for w in last_line.split() if w != "|"])
    if word_count > _MAX_FOOTER_LINE_WORDS:
        return None
    match = _TRAILING_PAGE_NUMBER.search(last_line)
    if not match:
        return None
    number = int(match.group(1))
    return number if number <= _MAX_PLAUSIBLE_PRINTED_PAGE else None


_MAX_SECTION_WORDS = 12


def _detect_section(text: str) -> str | None:
    """Best-effort guess at which section/heading a page belongs to.

    Takes the first non-empty line of the page's (already column-reordered)
    text as the section label, if it looks like a heading rather than a
    sentence of body text -- short, and not a formatted table row. This is a
    page-level approximation: a page with multiple sections on it (common in
    dense catalogs) only gets the first one, and headings that PyMuPDF merged
    into a longer descriptive block won't be picked up. It's a browsing aid
    for citations, not a guarantee -- None means "no confident guess."
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first_line = lines[0].strip()
    if "|" in first_line:  # a formatted table row, not a heading
        return None
    if len(first_line.split()) > _MAX_SECTION_WORDS:
        return None
    return first_line


def _format_table(rows: list[list[str | None]]) -> str:
    lines = []
    for row in rows:
        cells = [(cell or "").strip() for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _rects_overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty or a.get_area() == 0:
        return 0.0
    return inter.get_area() / a.get_area()


def _extract_page_text(page: fitz.Page) -> str:
    page_width = page.rect.width

    try:
        with contextlib.redirect_stdout(io.StringIO()):  # silence PyMuPDF's tips-to-console
            tables = page.find_tables().tables
    except Exception:
        tables = []

    table_bboxes = [fitz.Rect(t.bbox) for t in tables]

    items = []  # (x0, y0, text)
    for t, bbox in zip(tables, table_bboxes):
        formatted = _format_table(t.extract())
        if formatted:
            items.append((bbox.x0, bbox.y0, formatted))

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, _block_no, block_type = block
        if block_type != 0:  # skip image blocks
            continue
        text = text.strip()
        if not text:
            continue
        block_rect = fitz.Rect(x0, y0, x1, y1)
        # Skip text blocks that a detected table already covers, to avoid
        # duplicating (and re-scrambling) the same cells as loose words.
        if any(_rects_overlap_ratio(block_rect, tb) > 0.5 for tb in table_bboxes):
            continue
        items.append((x0, y0, text))

    # Bucket by left/right half of the page using each item's left edge, then
    # emit left-column content top-to-bottom before right-column content.
    # Single-column pages: everything starts near the left margin, so
    # everything buckets "left" and order is unchanged from plain top-to-
    # bottom reading order.
    left = sorted((i for i in items if i[0] < page_width / 2), key=lambda i: i[1])
    right = sorted((i for i in items if i[0] >= page_width / 2), key=lambda i: i[1])

    return "\n".join(text for _x0, _y0, text in left + right)


@dataclass
class PageText:
    page_number: int  # 1-indexed physical position in the PDF file
    text: str
    printed_page_number: int | None = None  # best-effort guess at the footer/printed number
    section: str | None = None  # best-effort guess at the page's heading/section


def _pages_from_doc(doc) -> list[PageText]:
    pages = []
    for index, page in enumerate(doc):
        text = _extract_page_text(page).strip()
        if text:
            pages.append(
                PageText(
                    page_number=index + 1,
                    text=text,
                    printed_page_number=_detect_printed_page_number(text),
                    section=_detect_section(text),
                )
            )
    return pages


def load_pdf_pages(pdf_path: str) -> list[PageText]:
    """Return one PageText per page of the PDF, skipping blank pages."""
    with fitz.open(pdf_path) as doc:
        return _pages_from_doc(doc)


def load_pdf_bytes(pdf_bytes: bytes) -> list[PageText]:
    """Same as load_pdf_pages but reads from in-memory bytes (Streamlit upload)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return _pages_from_doc(doc)
