"""Structured, append-only logging of each question the pipeline answers.

One JSON object per line (JSONL) in `data/logs/queries.jsonl`, so it can be
tailed, grepped, or loaded into pandas without a schema migration -- new
fields are additive. Kept separate from `src/evaluation.py`: that module
scores a fixed, hand-written case set; this one records what actually
happened in real usage (Streamlit, the API, ad-hoc scripts) with no ground
truth attached, for latency/error visibility and as raw material for a
future eval dashboard.

Never raises: a logging failure (disk full, bad permissions) must not break
answering a question, so every write is best-effort.
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("data/logs/queries.jsonl")

logger = logging.getLogger(__name__)


@dataclass
class QueryLogEntry:
    question: str
    provider: str
    top_k: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    search_query: str = ""
    rewritten: bool = False
    document_types: list[str] | None = None
    document_names: list[str] | None = None
    use_reranker: bool = True
    num_sources: int = 0
    retrieved_pages: list[int] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    grounded: bool | None = None
    error: str | None = None
    rewrite_ms: float | None = None
    retrieve_ms: float | None = None
    generate_ms: float | None = None
    total_ms: float | None = None


@contextmanager
def _timer(entry: QueryLogEntry, field_name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        setattr(entry, field_name, round((time.perf_counter() - start) * 1000, 1))


def write_entry(entry: QueryLogEntry) -> None:
    """Appends one entry as a JSON line. Swallows and logs any I/O error
    rather than propagating it -- see module docstring."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
    except Exception:
        # Broad on purpose: logging must never be why answering a question
        # fails, whatever the underlying cause (disk full, bad path, a
        # monkeypatched LOG_PATH in tests, permissions, ...).
        logger.warning("Failed to write query log entry", exc_info=True)


def read_entries(path: Path | str = LOG_PATH, limit: int | None = None) -> list[dict]:
    """Reads back logged entries, newest last. Missing file -> []. Any
    unparseable line is skipped rather than failing the whole read."""
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-limit:] if limit else entries
