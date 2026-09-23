"""Diagnostic tool: shows the full ranked retrieval list for a question,
so you can see exactly where the chunk you expected landed (or if it's
missing from the index entirely).

Usage:
    python scripts/debug_retrieval.py "What is the formula for Average Premium?"
    python scripts/debug_retrieval.py "..." --top-k 20
    python scripts/debug_retrieval.py "..." --page 12       # highlight matches on this page
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--page", type=int, default=None, help="highlight this page number")
    args = parser.parse_args()

    store = VectorStore()
    print(f"Indexed documents: {store.document_names()}")
    print(f"Total chunks in store: {store.count()}\n")

    matches = store.query(args.question, top_k=args.top_k)

    if not matches:
        print("No chunks returned. Is a document actually indexed?")
        return

    for i, m in enumerate(matches, start=1):
        marker = " <== target page" if args.page and m["display_page"] == args.page else ""
        page_note = f"page {m['display_page']}"
        if m["printed_page_number"] and m["printed_page_number"] != m["page_number"]:
            page_note += f" (file position: page {m['page_number']})"
        print(f"#{i:>2}  distance={m['distance']:.4f}  {m['document_name']}  {page_note}{marker}")
        snippet = m["text"][:160].replace("\n", " ")
        print(f"      {snippet}...\n")

    if args.page:
        pages_seen = {m["display_page"] for m in matches}
        if args.page not in pages_seen:
            print(f"Page {args.page} did not appear in the top {args.top_k} results at all.")
            print("This means either:")
            print("  1. The embedding for that chunk isn't semantically close to your question")
            print("     wording (try rephrasing, or this is a known limitation of fixed-size")
            print("     chunking + a small embedding model on tables/formulas).")
            print("  2. That page's text didn't extract cleanly from the PDF (scanned image,")
            print("     unusual table layout, etc.) - check with load_pdf_pages() directly.")


if __name__ == "__main__":
    main()
