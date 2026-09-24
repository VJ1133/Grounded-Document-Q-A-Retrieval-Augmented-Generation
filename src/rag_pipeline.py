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
import re
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

# Matches the structured trailer line the model is asked to append (see
# SYSTEM_PROMPT) -- e.g. "USED_PASSAGES: 1,3" -- so it can be parsed out and
# never shown to the user as raw text.
_USED_PASSAGES_RE = re.compile(r"(?im)^\s*USED_PASSAGES:\s*(.*?)\s*$")

SYSTEM_PROMPT = (
    "You are an insurance document assistant. Answer the user's question "
    "using ONLY the context passages provided below. "
    "If the context does not contain enough information to answer, say clearly: "
    f'"{INSUFFICIENT_CONTEXT_MESSAGE}" '
    "Do not use outside knowledge. Do not guess. Keep answers concise and factual. "
    "Tables are flattened to text. When a table header has several column "
    "groups (e.g. 'Earned Premiums | Earned Exposures' over '2022 2021 2020 "
    "2022 2021 2020'), the row values fill the groups IN ORDER: the first "
    "group's columns come first, the next group's after. Count columns "
    "carefully so each number is attributed to the right group and year -- "
    "never report a different measure than the one asked for. "
    "Whenever a figure comes from a table, name that table in your answer "
    "(its number/title and business type, e.g. 'Table 18A, Voluntary "
    "Business'). If the question does not specify a business type "
    "(Voluntary / Residual / Total) and the passages contain more than one, "
    "give the figures for EACH one found, labelled, instead of picking one. "
    "Do not include passage numbers or citation markers in your answer text -- "
    "sources are shown separately to the user, so just answer in plain prose.\n\n"
    "After your answer, on its own new line, output exactly one line in this "
    "exact format (this line is parsed by software, not shown to the user "
    "as-is, so it must match this format precisely and nothing else should "
    "follow it):\n"
    "USED_PASSAGES: <comma-separated passage numbers you actually relied on>\n"
    "If you gave the insufficient-context refusal, output USED_PASSAGES: none"
)


REWRITE_SYSTEM_PROMPT = (
    "You rewrite follow-up questions for a document search system. Given a "
    "conversation and a new question, output ONE standalone question that "
    "carries all the context needed to be understood without the "
    "conversation (resolve pronouns like 'it'/'that' and omitted subjects). "
    "If the new question is already standalone, output it unchanged. "
    "Output only the question text -- no quotes, no explanation, and do not "
    "answer it."
)

# How many prior (question, answer) turns are shown to the rewriter, and how
# much of each answer -- enough to resolve references, small enough to keep
# the prompt cheap and on-topic.
MAX_HISTORY_TURNS = 4
MAX_HISTORY_ANSWER_CHARS = 400


@dataclass
class RagAnswer:
    answer: str
    # What was actually searched for. Differs from the user's raw question
    # only when a follow-up was rewritten into a standalone question.
    search_query: str = ""
    sources: list[dict] = field(default_factory=list)
    # False whenever no real, source-backed answer was given (the model
    # explicitly declined, or there was nothing to search in the first
    # place). Set explicitly by answer_question rather than inferred, since
    # the "nothing to search" cases use different wording than the model's
    # own refusal message.
    grounded: bool = True


def _extract_used_passages(answer_text: str) -> tuple[str, set[int]]:
    """Strips the USED_PASSAGES trailer from the model's raw response and
    returns (clean_answer_text, {1-indexed passage numbers cited}).

    Returns an empty set (not an error) if the trailer is missing or
    unparseable -- some models occasionally drop instructed formatting, and
    "no highlighted sources" is a safe degradation, not a crash.
    """
    match = _USED_PASSAGES_RE.search(answer_text)
    if not match:
        return answer_text.strip(), set()

    clean_text = (answer_text[: match.start()] + answer_text[match.end() :]).strip()
    numbers = set()
    for token in match.group(1).split(","):
        token = token.strip()
        if token.isdigit():
            numbers.add(int(token))
    return clean_text, numbers


def _rewrite_followup(
    provider: str, question: str, history: list[tuple[str, str]]
) -> str:
    """Rewrites a follow-up into a standalone question using recent turns.

    Falls back to the original question on any failure or empty output, so a
    flaky rewrite never blocks answering (worst case: same behaviour as V3).
    """
    turns = history[-MAX_HISTORY_TURNS:]
    transcript = "\n".join(
        f"User: {q}\nAssistant: {a[:MAX_HISTORY_ANSWER_CHARS]}" for q, a in turns
    )
    prompt = (
        f"Conversation:\n{transcript}\n\n"
        f"New question: {question}\n\nStandalone question:"
    )
    try:
        rewritten = _PROVIDERS[provider](REWRITE_SYSTEM_PROMPT, prompt).strip()
    except Exception:
        return question
    rewritten = rewritten.splitlines()[0].strip().strip('"') if rewritten else ""
    return rewritten or question


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
    history: list[tuple[str, str]] | None = None,
) -> RagAnswer:
    """`history` is prior (question, answer) turns, oldest first. When given,
    the question is first rewritten into a standalone query so retrieval
    works for follow-ups like "what about for Texas?"."""
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")

    search_query = _rewrite_followup(provider, question, history) if history else question

    matches = vector_store.query(
        search_query, top_k=top_k, document_types=document_types, document_names=document_names
    )

    if not matches:
        if (document_types or document_names) and vector_store.count() > 0:
            message = (
                "No documents match the current filter, so there is nothing to search. "
                "Clear or adjust the document filter and try again."
            )
        else:
            message = "No documents have been uploaded yet, so there is nothing to search."
        return RagAnswer(answer=message, search_query=search_query, sources=[], grounded=False)

    context = _build_context(matches)
    user_prompt = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    raw_answer_text = _PROVIDERS[provider](SYSTEM_PROMPT, user_prompt)
    answer_text, cited_passage_numbers = _extract_used_passages(raw_answer_text)
    grounded = INSUFFICIENT_CONTEXT_MESSAGE.lower() not in answer_text.lower()

    # Passage numbers in the prompt/response are 1-indexed and match the
    # order of `matches` -- mark each source so the UI can highlight
    # specifically which passage(s) the model says it actually used,
    # instead of leaving every retrieved passage looking equally relevant.
    for i, m in enumerate(matches, start=1):
        m["cited"] = i in cited_passage_numbers

    return RagAnswer(
        answer=answer_text, search_query=search_query, sources=matches, grounded=grounded
    )
