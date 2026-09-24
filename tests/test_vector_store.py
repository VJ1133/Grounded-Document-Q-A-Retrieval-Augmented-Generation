from src.chunker import Chunk
from src.vector_store import VectorStore


def _make_store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma_test"))


def _chunk(text, page=1, index=0, section=None):
    return Chunk(text=text, page_number=page, chunk_index=index, section=section)


def test_add_and_query_single_document(tmp_path):
    store = _make_store(tmp_path)
    store.add_document(
        "policy_a.pdf",
        [_chunk("The deductible for collision coverage is 500 dollars.")],
        document_type="Policy",
    )

    matches = store.query("What is the collision deductible?", top_k=3)

    assert len(matches) == 1
    assert matches[0]["document_name"] == "policy_a.pdf"
    assert matches[0]["document_type"] == "Policy"


def test_query_searches_across_multiple_documents(tmp_path):
    store = _make_store(tmp_path)
    store.add_document(
        "policy_a.pdf", [_chunk("Collision deductible is 500 dollars.")], document_type="Policy"
    )
    store.add_document(
        "catalog_b.pdf",
        [_chunk("Ratchet handle A715 costs 612 rupees.")],
        document_type="Catalog",
    )

    matches = store.query("ratchet handle price", top_k=5)

    assert any(m["document_name"] == "catalog_b.pdf" for m in matches)


def test_query_filters_by_document_type(tmp_path):
    store = _make_store(tmp_path)
    store.add_document(
        "policy_a.pdf", [_chunk("Collision deductible is 500 dollars.")], document_type="Policy"
    )
    store.add_document(
        "catalog_b.pdf", [_chunk("Ratchet handle costs 612 rupees.")], document_type="Catalog"
    )

    matches = store.query("deductible", top_k=5, document_types=["Catalog"])

    assert all(m["document_type"] == "Catalog" for m in matches)
    assert all(m["document_name"] == "catalog_b.pdf" for m in matches)


def test_query_filters_by_document_name(tmp_path):
    store = _make_store(tmp_path)
    store.add_document("policy_a.pdf", [_chunk("Collision deductible is 500 dollars.")])
    store.add_document("policy_b.pdf", [_chunk("Collision deductible is 750 dollars.")])

    matches = store.query("deductible", top_k=5, document_names=["policy_b.pdf"])

    assert all(m["document_name"] == "policy_b.pdf" for m in matches)


def test_document_types_lists_distinct_types(tmp_path):
    store = _make_store(tmp_path)
    store.add_document("a.pdf", [_chunk("text one")], document_type="Policy")
    store.add_document("b.pdf", [_chunk("text two")], document_type="Catalog")
    store.add_document("c.pdf", [_chunk("text three")], document_type="Policy")

    assert store.document_types() == ["Catalog", "Policy"]


def test_document_summaries_counts_chunks_per_document(tmp_path):
    store = _make_store(tmp_path)
    store.add_document(
        "a.pdf", [_chunk("one", index=0), _chunk("two", index=1)], document_type="Policy"
    )
    store.add_document("b.pdf", [_chunk("three", index=0)], document_type="Catalog")

    summaries = {s["document_name"]: s for s in store.document_summaries()}

    assert summaries["a.pdf"]["chunk_count"] == 2
    assert summaries["a.pdf"]["document_type"] == "Policy"
    assert summaries["b.pdf"]["chunk_count"] == 1


def test_delete_document_removes_only_that_document(tmp_path):
    store = _make_store(tmp_path)
    store.add_document("a.pdf", [_chunk("keep this one")])
    store.add_document("b.pdf", [_chunk("delete this one")])

    store.delete_document("b.pdf")

    assert store.document_names() == ["a.pdf"]
    assert store.count() == 1


def test_default_document_type_when_not_specified(tmp_path):
    store = _make_store(tmp_path)
    store.add_document("a.pdf", [_chunk("no type given")])

    matches = store.query("no type given", top_k=1)

    assert matches[0]["document_type"] == "General"


def test_hybrid_retrieval_surfaces_exact_term_diluted_by_similar_chunks(tmp_path):
    """Regression test for the real bug this fixed: a dense list of many
    near-duplicate state notes (differing mainly by state name) buries the
    one containing the actual answer under pure semantic search, because
    the chunk's overall embedding is dominated by the shared boilerplate
    rather than the one distinguishing term. BM25 keyword matching on the
    exact term (via Reciprocal Rank Fusion) should still surface it.
    """
    store = _make_store(tmp_path)

    # Many near-duplicate chunks sharing the same boilerplate structure and
    # vocabulary, differing only by state name -- mirrors the real
    # multi-state PIP notes list that diluted semantic search.
    filler_states = [
        "Alabama", "Georgia", "Idaho", "Kansas", "Maine", "Nevada",
        "Ohio", "Oregon", "Tennessee", "Vermont", "Wyoming", "Montana",
    ]
    chunks = [
        _chunk(f"{state} has no special minimum requirement for PIP coverage.", index=i)
        for i, state in enumerate(filler_states)
    ]
    chunks.append(
        _chunk(
            "Utah has a special minimum requirement: there is a $3,000 minimum for PIP.",
            index=len(filler_states),
        )
    )
    store.add_document("state_notes.pdf", chunks)

    matches = store.query("what is the minimum PIP for Utah?", top_k=3)

    assert any("Utah" in m["text"] and "3,000" in m["text"] for m in matches)
