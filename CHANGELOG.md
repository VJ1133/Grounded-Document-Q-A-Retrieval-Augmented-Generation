# Changelog

## V3 — Citations & Grounding

- `RagAnswer.grounded`: an explicit boolean, set by `answer_question()`
  rather than inferred by the UI, so "was this answer actually supported"
  is a real piece of data, not just prose the model happened to write.
- Grounded/ungrounded badge in the UI (✓ Grounded / ⚠ Not supported by
  documents), with retrieved evidence still shown either way — so an
  unsupported answer still lets you see *why* (the closest matches just
  weren't good enough).
- Retrieval scores (passage distances) are now opt-in via a "Show retrieval
  scores (debug)" checkbox instead of always shown.
- Tightened the generation prompt to keep the model's answer prose free of
  its own inline citation markers, since sources are always rendered
  separately by the UI — avoids duplicate/inconsistent citation styles.
- `scripts/test_grounding.py`: a curated, repeatable grounding test set
  (some answerable questions, some deliberately out-of-scope) run against
  live documents and a real LLM call — verified 7/7 passing against Groq.
- New `tests/test_rag_pipeline.py` (5 tests) covering the `grounded` logic
  itself via a monkeypatched provider, without needing a live LLM call.

**What I learned:** most of V3's spec was already satisfied by V1/V2
hardening (page citations, evidence panel, a strict grounded prompt) — the
part that was actually missing was turning "the model said it doesn't
know" from implicit prose into an explicit, testable boolean. That's a
small code change but it's the difference between a UI that *looks*
trustworthy and one where trustworthiness is actually verifiable.

## V1 — Basic PDF RAG

- Single-PDF upload via Streamlit.
- PyMuPDF page-level text extraction.
- Fixed-size, per-page, overlapping word chunking.
- Local embeddings via Sentence Transformers (`all-MiniLM-L6-v2`).
- Persistent local vector storage via ChromaDB.
- Top-k retrieval and grounded generation via a local Ollama model
  (`gemma3:4b`), with an explicit "insufficient context" fallback.
- Custom dark/gold Streamlit theme (hero header, glass cards, source chips,
  status indicators) built entirely with Streamlit + CSS — no separate
  frontend framework.

**What I learned:** most of the perceived quality of a RAG answer comes from
prompt discipline ("answer only from context, say when you don't know") and
from keeping page numbers attached to every chunk all the way through
retrieval — losing that thread anywhere in the pipeline makes citations
impossible to reconstruct later.

## V1 hardening (bugs found testing with real documents)

- Fixed multi-column/table content being extracted in the wrong reading
  order (PyMuPDF's plain text mode follows content-stream order, not visual
  layout), which had misattributed a product row to the wrong section
  header in a real catalog PDF. Fixed via `find_tables()` + left/right
  column bucketing in `src/document_loader.py`.
- Added a printed-page-number detector so citations show the number
  actually stamped on the page instead of raw physical file position —
  tightened after it initially misfired on table rows ending in large
  prices.
- Added a switchable local (Ollama)/hosted (Groq API, `openai/gpt-oss-120b`)
  generation provider — the local 4B model was unreliable at precise
  lookups in dense tabular context; Groq's larger free-tier model fixed it.

**What I learned:** a synthetic demo PDF will never surface the bugs a real,
messy document will. The table/column extraction bug, the page-number
offset bug, and the small-model lookup failure were all found by testing
with an actual multi-column industrial price list, not by writing more
unit tests against clean synthetic input.

## V2 — Multi-Document RAG

- Multi-file upload: the uploader accepts several PDFs at once, each
  ingested and embedded independently into the same shared collection.
- Document type tagging at upload time (free-text, user-controlled).
- Metadata filtering: scope a question to selected document types and/or
  specific documents instead of always searching everything.
- Per-document management: remove a single indexed document without
  clearing the whole collection.
- Best-effort section/heading detection per page, surfaced in citations.
- New `tests/test_vector_store.py` covering cross-document search, type/name
  filtering, per-document deletion, and document summaries.

**What I learned:** cross-document search mostly "fell out" of V1's design
for free — ChromaDB already searches its whole collection regardless of how
many documents are in it. The actual V2 work was in the parts V1 never
needed: giving each chunk enough metadata to *filter* on, and giving the UI
a way to manage a growing collection instead of only "upload one, clear
all."
