import json

from src.query_log import QueryLogEntry, read_entries, write_entry


def test_write_entry_appends_one_json_line(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "queries.jsonl"
    monkeypatch.setattr("src.query_log.LOG_PATH", log_path)

    write_entry(QueryLogEntry(question="a", provider="ollama", top_k=8))
    write_entry(QueryLogEntry(question="b", provider="groq", top_k=8))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "a"
    assert json.loads(lines[1])["question"] == "b"


def test_write_entry_never_raises_on_bad_path(monkeypatch):
    # a path with a NUL-ish/invalid parent can't be created; write_entry
    # must swallow the error rather than propagate it (see module docstring)
    monkeypatch.setattr("src.query_log.LOG_PATH", None)

    write_entry(QueryLogEntry(question="a", provider="ollama", top_k=8))  # must not raise


def test_read_entries_missing_file_returns_empty_list(tmp_path):
    assert read_entries(tmp_path / "nope.jsonl") == []


def test_read_entries_skips_unparseable_lines(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"question": "a"}\nnot json\n{"question": "b"}\n', encoding="utf-8")

    assert [e["question"] for e in read_entries(path)] == ["a", "b"]


def test_read_entries_respects_limit(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(json.dumps({"question": str(i)}) for i in range(5)), encoding="utf-8")

    assert [e["question"] for e in read_entries(path, limit=2)] == ["3", "4"]
