from retrievalops import Chunk, EvidenceGroundedRAG, ResponseState


def corpus():
    return [
        Chunk("approval", "travel", "Approval", "Emergency travel requires director approval.", "v2"),
        Chunk("receipts", "travel", "Expenses", "Receipts are required for travel expenses.", "v2"),
        Chunk("retired", "old-travel", "Approval", "Emergency travel requires manager approval.", "v1", "retired"),
    ]


def test_pipeline_returns_grounded_answer_with_trace():
    response = EvidenceGroundedRAG(corpus()).answer("Who approves emergency travel?", requested_version="v2")

    assert response.state is ResponseState.ANSWER
    assert response.answer == "Emergency travel requires director approval."
    assert response.claim_trace[0]["chunk_id"] == "approval"
    assert response.claim_trace[0]["section"] == "Approval"
    assert response.evidence[0].chunk.version == "v2"


def test_inactive_and_wrong_version_evidence_never_reaches_answer():
    response = EvidenceGroundedRAG(corpus()).answer("Who approves emergency travel?", requested_version="v9")

    assert response.state is ResponseState.RETRY
    assert response.answer is None


def test_repeat_after_insufficient_evidence_abstains():
    response = EvidenceGroundedRAG(corpus()).answer("unrelated astrophysics", retry_attempt=1)

    assert response.state is ResponseState.ABSTAIN
    assert response.answer is None
