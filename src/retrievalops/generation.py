"""Grounded extractive response construction."""

from __future__ import annotations

from .schema import Evidence, Response, ResponseState


def build_response(
    state: ResponseState,
    evidence: tuple[Evidence, ...],
    reason: str,
) -> Response:
    """Construct an answer only when the reliability layer permits it."""
    if state is not ResponseState.ANSWER:
        return Response(state=state, answer=None, evidence=evidence, reason=reason)
    if not evidence:
        raise ValueError("ANSWER requires selected evidence")
    support = evidence[0]
    claim = support.chunk.text
    trace = (
        {
            "chunk_id": support.chunk.chunk_id,
            "document_id": support.chunk.document_id,
            "section": support.chunk.section,
            "claim": claim,
        },
    )
    return Response(
        state=ResponseState.ANSWER,
        answer=claim,
        evidence=(support,),
        claim_trace=trace,
        reason=reason,
    )
