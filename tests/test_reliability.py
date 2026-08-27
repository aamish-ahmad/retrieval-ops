from retrievalops.reliability import decide_state
from retrievalops.schema import Chunk, Evidence, ResponseState


def evidence(chunk_id, document_id, score=0.8):
    return Evidence(
        Chunk(chunk_id, document_id, "Scope", "Shared rule.", "v1"),
        sparse_score=score,
        dense_score=score,
        fusion_score=score,
        rerank_score=score,
    )


def test_equal_top_documents_require_clarification():
    state, selected, _ = decide_state([evidence("one", "a"), evidence("two", "b")])

    assert state is ResponseState.CLARIFY
    assert len(selected) == 2


def test_weak_evidence_retries_before_abstaining():
    weak = evidence("one", "a", score=0.01)

    assert decide_state([weak])[0] is ResponseState.RETRY
    assert decide_state([weak], retry_attempt=1)[0] is ResponseState.ABSTAIN
