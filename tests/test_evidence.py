from retrievalops.evidence import select_evidence
from retrievalops.models import Chunk, Evidence, ResponseState


def test_rejects_wrong_version_and_stale_evidence():
    items = [
        Evidence(Chunk("a", "doc", "s", "old", version="v1", effective_state="active"), 1.0),
        Evidence(Chunk("b", "doc", "s", "stale", version="v2", effective_state="stale"), 0.9),
    ]
    state, selected, _ = select_evidence(items, target_version="v2", retry_attempt=1)
    assert state is ResponseState.ABSTAIN
    assert selected == []


def test_clarifies_exact_cross_document_tie():
    items = [
        Evidence(Chunk("a", "doc-a", "s", "one"), 1.0),
        Evidence(Chunk("b", "doc-b", "s", "two"), 1.0),
    ]
    state, selected, _ = select_evidence(items)
    assert state is ResponseState.CLARIFY
    assert len(selected) == 2
