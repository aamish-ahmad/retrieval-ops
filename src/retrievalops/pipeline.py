"""End-to-end evidence-grounded RAG pipeline."""

from __future__ import annotations

from .generation import build_response
from .reliability import decide_state
from .retrieval import HybridRetriever
from .schema import Chunk, Response


class EvidenceGroundedRAG:
    """Compose retrieval, evidence control, and grounded response construction."""

    def __init__(self, chunks: list[Chunk], *, minimum_evidence_score: float = 0.05) -> None:
        self.retriever = HybridRetriever(chunks)
        self.minimum_evidence_score = minimum_evidence_score

    def answer(
        self,
        question: str,
        *,
        requested_version: str | None = None,
        retry_attempt: int = 0,
    ) -> Response:
        evidence = self.retriever.retrieve(question)
        state, selected, reason = decide_state(
            evidence,
            requested_version=requested_version,
            retry_attempt=retry_attempt,
            minimum_evidence_score=self.minimum_evidence_score,
        )
        return build_response(state, selected, reason)
