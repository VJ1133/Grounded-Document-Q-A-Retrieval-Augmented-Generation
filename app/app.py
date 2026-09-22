"""Insurance AI Knowledge Assistant — V1 Streamlit UI.

Upload a PDF, ask a question, get an answer grounded in that PDF using a
local embedding model, ChromaDB, and a local Ollama LLM.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_pages
from src.document_loader import load_pdf_bytes
from src.rag_pipeline import OLLAMA_MODEL, answer_question
from src.vector_store import VectorStore

st.set_page_config(
    page_title="Insurance AI Knowledge Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- resources

@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    return VectorStore()


def ollama_is_reachable() -> bool:
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


store = get_vector_store()

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, RagAnswer)


# --------------------------------------------------------------------- hero

doc_count = len(store.document_names())
chunk_count = store.count()
ollama_ok = ollama_is_reachable()

st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-kicker"><span class="dot"></span> LOCAL &nbsp;·&nbsp; PRIVATE &nbsp;·&nbsp; GROUNDED</div>
        <div class="hero-title">Insurance AI Knowledge Assistant</div>
        <div class="hero-sub">
            Upload insurance PDFs and ask questions in plain English. Every answer is
            retrieved from your own documents with a local embedding model, a local
            vector database, and a local LLM — nothing leaves your machine.
        </div>
        <div class="hero-stats">
            <div class="hero-stat"><div class="num">{doc_count}</div><div class="label">Documents indexed</div></div>
            <div class="hero-stat"><div class="num">{chunk_count}</div><div class="label">Chunks embedded</div></div>
            <div class="hero-stat"><div class="num">{OLLAMA_MODEL}</div><div class="label">Local model</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)
    led = "led-green" if ollama_ok else "led-red"
    label = "Ollama connected" if ollama_ok else "Ollama unreachable"
    st.markdown(
        f'<div class="status-pill"><span class="led {led}"></span>{label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="status-pill"><span class="led led-green"></span>Embedding model ready</div>',
        unsafe_allow_html=True,
    )
    if not ollama_ok:
        st.caption(f"Run `ollama serve` and `ollama pull {OLLAMA_MODEL}` first.")

    st.markdown('<div class="section-label">Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        already_indexed = store.has_document(uploaded_file.name)
        if already_indexed:
            st.info(f"'{uploaded_file.name}' is already indexed.")
        elif st.button("Ingest document", type="primary", use_container_width=True):
            with st.spinner("Extracting text, chunking, and embedding..."):
                pages = load_pdf_bytes(uploaded_file.getvalue())
                chunks = chunk_pages(pages)
                store.add_document(uploaded_file.name, chunks)
            st.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages.")
            st.rerun()

    st.markdown('<div class="section-label">Indexed Documents</div>', unsafe_allow_html=True)
    names = store.document_names()
    if not names:
        st.caption("No documents uploaded yet.")
    else:
        for name in names:
            st.markdown(f'<div class="doc-pill">📄 {name}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear all documents", use_container_width=True):
        store.reset()
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------------------------- main

st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

with st.form("ask_form", clear_on_submit=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Question",
            placeholder="e.g. What exposure characteristics does the catastrophe model use?",
            label_visibility="collapsed",
        )
    with col2:
        ask_clicked = st.form_submit_button(
            "Ask →", type="primary", use_container_width=True
        )

if ask_clicked and question.strip():
    if doc_count == 0:
        st.warning("Upload a PDF first — there is nothing to search yet.")
    elif not ollama_ok:
        st.error(f"Ollama is not reachable. Run `ollama serve` in a terminal, then try again.")
    else:
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            result = answer_question(store, question.strip())
        st.session_state.history.insert(0, (question.strip(), result))

st.markdown("<br>", unsafe_allow_html=True)

for q, result in st.session_state.history:
    st.markdown(f"**{q}**")
    st.markdown(f'<div class="answer-card">{result.answer}</div>', unsafe_allow_html=True)

    if result.sources:
        chips = "".join(
            f'<span class="source-chip">📄 {s["document_name"]} · p.{s["page_number"]}</span>'
            for s in result.sources
        )
        st.markdown(f"<div style='margin-top:10px'>{chips}</div>", unsafe_allow_html=True)

        with st.expander("View retrieved evidence"):
            for i, s in enumerate(result.sources, start=1):
                st.markdown(
                    f"**Passage {i} — {s['document_name']}, page {s['page_number']}** "
                    f"(distance: {s['distance']:.3f})"
                )
                st.caption(s["text"])
    st.markdown("<hr style='border-color: rgba(201,162,75,0.12)'>", unsafe_allow_html=True)

st.markdown(
    '<div class="app-footer">Insurance AI Knowledge Assistant · V1 Basic RAG · '
    "Runs 100% locally — no data leaves this machine.</div>",
    unsafe_allow_html=True,
)
