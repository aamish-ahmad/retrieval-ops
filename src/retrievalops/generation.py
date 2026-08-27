from __future__ import annotations

from .models import Evidence, Response, ResponseState


def generate_extractive(state: ResponseState, evidence: list[Evidence], reason: str) -> Response:
    """Return an inspectable extractive answer, or no answer for non-answer states."""
    if state is not ResponseState.ANSWER:
        return Response(state=state, answer="", evidence=tuple(evidence), reason=reason)
    if not evidence:
        raise ValueError("ANSWER state requires evidence")
    support = evidence[0]
    return Response(
        state=ResponseState.ANSWER,
        answer=support.chunk.text,
        evidence=(support,),
        reason=reason,
    )
