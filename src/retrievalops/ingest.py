"""Validated JSONL ingestion for local corpora."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Chunk

_REQUIRED = ("chunk_id", "document_id", "section", "text")


def load_jsonl(path: str | Path) -> list[Chunk]:
    """Load one JSON object per line while preserving provenance metadata."""
    source = Path(path)
    rows: list[Chunk] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        for field in _REQUIRED:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"line {line_number} requires non-empty {field}")
        chunk_id = row["chunk_id"]
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        seen.add(chunk_id)
        version = row.get("version")
        if version is not None and not isinstance(version, str):
            raise ValueError(f"line {line_number} version must be a string or null")
        effective_state = row.get("effective_state", "active")
        if not isinstance(effective_state, str) or not effective_state.strip():
            raise ValueError(f"line {line_number} effective_state must be a non-empty string")
        rows.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                section=row["section"],
                text=row["text"],
                version=version,
                effective_state=effective_state,
            )
        )
    if not rows:
        raise ValueError("corpus is empty")
    return rows
