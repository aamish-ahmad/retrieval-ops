import json

import pytest

from retrievalops.ingest import load_jsonl


def test_load_jsonl_preserves_provenance(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({
        "chunk_id": "one", "document_id": "policy", "section": "Scope", "text": "Approved text."
    }) + "\n", encoding="utf-8")

    [chunk] = load_jsonl(corpus)

    assert (chunk.chunk_id, chunk.document_id, chunk.section) == ("one", "policy", "Scope")


def test_load_jsonl_rejects_duplicate_identifiers(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    row = {"chunk_id": "one", "document_id": "policy", "section": "Scope", "text": "Approved text."}
    corpus.write_text("\n".join([json.dumps(row), json.dumps(row)]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(corpus)
