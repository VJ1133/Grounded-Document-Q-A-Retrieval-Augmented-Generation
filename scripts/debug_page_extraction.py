"""Diagnostic: scans a PDF page-by-page and flags pages that likely lost
content to image-based rendering (scanned pages, table screenshots, charts).

A page is flagged when it has very little extracted text but DOES contain
one or more embedded images -- a strong signal that meaningful content
(a table, a scanned form, a chart) exists on that page as pixels, not as
extractable text, and so never made it into the RAG index at all.

Usage:
    python scripts/debug_page_extraction.py data/sample_documents/your_file.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pymupdf as fitz

TEXT_LENGTH_THRESHOLD = 40  # chars; below this, a page is "text-light"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    args = parser.parse_args()

    doc = fitz.open(args.pdf_path)
    flagged = []

    for index, page in enumerate(doc):
        text = page.get_text("text").strip()
        image_count = len(page.get_images())
        physical_page = index + 1

        status = "ok"
        if len(text) < TEXT_LENGTH_THRESHOLD and image_count > 0:
            status = "FLAGGED: text-light + has images (likely scanned/image content)"
            flagged.append(physical_page)
        elif len(text) == 0:
            status = "FLAGGED: no extractable text at all"
            flagged.append(physical_page)

        print(f"page {physical_page:>3}  chars={len(text):>5}  images={image_count:>2}  {status}")

    doc.close()

    print()
    if flagged:
        print(f"{len(flagged)} page(s) likely lost content to image-based rendering: {flagged}")
        print("These pages were skipped entirely (0 chars) or indexed with a content gap")
        print("(text-light) by the current PyMuPDF-only extraction. OCR would be required")
        print("to recover their content.")
    else:
        print("No pages flagged. Extraction looks complete for this document.")


if __name__ == "__main__":
    main()
