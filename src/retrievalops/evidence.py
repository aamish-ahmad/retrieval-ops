from __future__ import annotations

from .models import Evidence, ResponseState


def select_evidence(
    candidates: list[Evidence],
    *,
    target_version: str | None = None,
    retry_attempt: int = 0,
) -> tuple[ResponseState, list[Evidence], str]:
    """Apply simple, explicit admissibility rules before generation."""
    admissible: list[Evidence] = []
    for item in candidates:
        chunk = item.chunk
        if target_version is not None and chunk.version not in {None, target_version}:
            continue
        if chunk.effective_state not in {None, "active"}:
            continue
        admissible.append(item)

    if not admissible:
        if retry_attempt < 1:
            return ResponseState.RETRY, [], "no admissible evidence; one retry available"
        return ResponseState.ABSTAIN, [], "no admissible evidence after retry"

    if len(admissible) > 1:
        first, second = admissible[:2]
        if first.score == second.score and first.chunk.document != second.chunk.document:
            return ResponseState.CLARIFY, admissible, "top evidence is tied across documents"

    return ResponseState.ANSWER, admissible, "admissible evidence selected"
