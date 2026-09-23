"""Ties retrieval and generation together into a single answer() call.

Generation is pluggable between two providers:
  - "ollama": a local model via Ollama. Free, fully private, but limited by
    whatever model fits on your machine's RAM (small models struggle with
    precise lookups in dense, table-heavy context).
  - "groq": a free hosted API (openai/gpt-oss-120b). Requires a GROQ_API_KEY
    and sends retrieved passage text to Groq's servers for that call -- a
    real privacy tradeoff in exchange for a much larger, more capable model.

Embeddings and vector search (src/embeddings.py, src/vector_store.py) always
run locally regardless of which generation provider is selected.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.vector_store import VectorStore

load_dotenv()  # reads GROQ_API_KEY etc. from a local .env file, if present

OLLAMA_MODEL = "gemma3:4b"
GROQ_MODEL = "openai/gpt-oss-120b"

# Returned verbatim by the model (and by answer_question's own early exits)
# whenever the retrieved context can't support an answer. Matching on this
# exact string is how RagAnswer.grounded is computed below -- it must stay
# in sync with what the prompt asks the model to say.
INSUFFICIENT_CONTEXT_MESSAGE = (
    "The uploaded document(s) do not contain enough information to answer this question."
)

SYSTEM_PROMPT = (
    "You are an insurance document assistant. Answer the user's question "
    "using ONLY the context passages provided below. "
    "If the context does not contain enough information to answer, say clearly: "
    f'"{INSUFFICIENT_CONTEXT_MESSAGE}" '
    "Do not use outside knowledge. Do not guess. Keep answers concise and factual. "
    "Do not include passage numbers or citation markers in your answer text -- "
    "sources are shown separately to the user, so just answer in plain prose."
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    # False whenever no real, source-backed answer was given (the model
    # explicitly declined, or there was nothing to search in the first
    # place). Set explicitly by answer_question rather than inferred, since
    # the "nothing to search" cases use different wording than the model's
    # own refusal message.
    grounded: bool = True


def _build_context(matches: list[dict]) -> str:
    blocks = []
    for i, m in enumerate(matches, start=1):
        section_note = f" | section: {m['section']}" if m.get("section") else ""
        blocks.append(
            f"[Passage {i} | {m['document_name']} | page {m['display_page']}{section_note}]\n"
            f"{m['text']}"
        )
    return "\n\n".join(blocks)


def _generate_with_ollama(system_prompt: str, user_prompt: str) -> str:
    import ollama

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def _generate_with_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at console.groq.com and add it "
            "to a .env file (GROQ_API_KEY=...) or your environment variables."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


_PROVIDERS = {
    "ollama": _generate_with_ollama,
    "groq": _generate_with_groq,
}


def answer_question(
    vector_store: VectorStore,
    question: str,
    top_k: int = 8,
    provider: str = "ollama",
    document_types: list[str] | None = None,
    document_names: list[str] | None = None,
) -> RagAnswer:
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")

    matches = vector_store.query(
        question, top_k=top_k, document_types=document_types, document_names=document_names
    )

    if not matches:
        if (document_types or document_names) and vector_store.count() > 0:
            message = (
                "No documents match the current filter, so there is nothing to search. "
                "Clear or adjust the document filter and try again."
            )
        else:
            message = "No documents have been uploaded yet, so there is nothing to search."
        return RagAnswer(answer=message, sources=[], grounded=False)

    context = _build_context(matches)
    user_prompt = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    answer_text = _PROVIDERS[provider](SYSTEM_PROMPT, user_prompt)
    grounded = INSUFFICIENT_CONTEXT_MESSAGE.lower() not in answer_text.lower()

    return RagAnswer(answer=answer_text, sources=matches, grounded=grounded)
