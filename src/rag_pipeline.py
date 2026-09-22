"""Ties retrieval and local generation together into a single answer() call."""

from dataclasses import dataclass, field

import ollama

from src.vector_store import VectorStore

OLLAMA_MODEL = "gemma3:4b"

SYSTEM_PROMPT = (
    "You are an insurance document assistant. Answer the user's question "
    "using ONLY the context passages provided below. "
    "If the context does not contain enough information to answer, say clearly: "
    "\"The uploaded document(s) do not contain enough information to answer this question.\" "
    "Do not use outside knowledge. Do not guess. Keep answers concise and factual."
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)


def _build_context(matches: list[dict]) -> str:
    blocks = []
    for i, m in enumerate(matches, start=1):
        blocks.append(
            f"[Passage {i} | {m['document_name']} | page {m['page_number']}]\n{m['text']}"
        )
    return "\n\n".join(blocks)


def answer_question(vector_store: VectorStore, question: str, top_k: int = 5) -> RagAnswer:
    matches = vector_store.query(question, top_k=top_k)

    if not matches:
        return RagAnswer(
            answer="No documents have been uploaded yet, so there is nothing to search.",
            sources=[],
        )

    context = _build_context(matches)
    user_prompt = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and mention which passage(s) support your answer."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer_text = response["message"]["content"]

    return RagAnswer(answer=answer_text, sources=matches)
