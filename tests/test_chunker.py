from src.chunker import chunk_pages
from src.document_loader import PageText


def test_chunk_pages_single_short_page():
    pages = [PageText(page_number=1, text="one two three four five")]
    chunks = chunk_pages(pages, chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == "one two three four five"


def test_chunk_pages_splits_long_page_with_overlap():
    words = [f"w{i}" for i in range(25)]
    pages = [PageText(page_number=3, text=" ".join(words))]
    chunks = chunk_pages(pages, chunk_size=10, overlap=3)

    assert len(chunks) == 4
    assert all(c.page_number == 3 for c in chunks)
    # overlap: last word of chunk N reappears near the start of chunk N+1
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]


def test_chunk_pages_skips_empty_pages():
    pages = [
        PageText(page_number=1, text=""),
        PageText(page_number=2, text="hello world"),
    ]
    chunks = chunk_pages(pages, chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_pages_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages([PageText(page_number=1, text="a b c")], chunk_size=5, overlap=5)
