"""Explicit evidence-admissibility and safe response-state decisions."""

from __future__ import annotations

from .schema import Evidence, ResponseState

_ELIGIBLE_STATES = frozenset({"active", "current"})


def decide_state(
    evidence: list[Evidence],
    *,
    requested_version: str | None = None,
    retry_attempt: int = 0,
    minimum_evidence_score: float = 0.05,
) -> tuple[ResponseState, tuple[Evidence, ...], str]:
    """Select usable evidence before answer construction is allowed."""
    eligible = [
        item
        for item in evidence
        if item.chunk.effective_state.casefold() in _ELIGIBLE_STATES
        and (requested_version is None or item.chunk.version == requested_version)
    ]
    if not eligible:
        state = ResponseState.ABSTAIN if retry_attempt else ResponseState.RETRY
        return state, (), "no current evidence matches the requested scope"
    if (
        eligible[0].rerank_score < minimum_evidence_score
        or (eligible[0].sparse_score <= 0 and eligible[0].dense_score <= 0)
    ):
        state = ResponseState.ABSTAIN if retry_attempt else ResponseState.RETRY
        return state, (), "retrieved evidence is too weak to support an answer"
    if len(eligible) > 1:
        first, second = eligible[:2]
        if first.rerank_score == second.rerank_score and first.chunk.document_id != second.chunk.document_id:
            return ResponseState.CLARIFY, tuple(eligible[:2]), "top evidence is tied across documents"
    return ResponseState.ANSWER, tuple(eligible), "admissible evidence selected"
