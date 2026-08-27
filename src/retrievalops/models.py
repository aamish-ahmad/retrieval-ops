from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    section: str
    text: str
    source: str | None = None
    version: str | None = None
    effective_state: str | None = None


@dataclass(frozen=True)
class Evidence:
    chunk: Chunk
    score: float


class ResponseState(str, Enum):
    ANSWER = "ANSWER"
    CLARIFY = "CLARIFY"
    RETRY = "RETRY"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Response:
    state: ResponseState
    answer: str
    evidence: tuple[Evidence, ...]
    reason: str
