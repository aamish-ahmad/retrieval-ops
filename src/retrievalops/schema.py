"""Stable data structures for inspectable RAG answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ResponseState(StrEnum):
    ANSWER = "ANSWER"
    CLARIFY = "CLARIFY"
    RETRY = "RETRY"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Chunk:
    """A retrievable corpus segment with provenance and lifecycle metadata."""

    chunk_id: str
    document_id: str
    section: str
    text: str
    version: str | None = None
    effective_state: str = "active"


@dataclass(frozen=True)
class Evidence:
    """A ranked source segment and the scores that selected it."""

    chunk: Chunk
    sparse_score: float
    dense_score: float
    fusion_score: float
    rerank_score: float


@dataclass(frozen=True)
class Response:
    """A grounded response plus its inspectable source-to-claim trace."""

    state: ResponseState
    answer: str | None
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    claim_trace: tuple[dict[str, str], ...] = field(default_factory=tuple)
    reason: str | None = None
