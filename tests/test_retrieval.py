from retrievalops import Chunk, RetrievalPipeline, ResponseState
from retrievalops.bm25 import BM25Index
from retrievalops.fusion import reciprocal_rank_fusion
from retrievalops.models import Evidence
from retrievalops.rerank import rerank


def corpus():
    return [
        Chunk("c1", "policy-a", "leave", "Employees receive twenty days of annual leave.", source="handbook"),
        Chunk("c2", "policy-b", "travel", "Travel expenses require manager approval before booking.", source="handbook"),
        Chunk("c3", "policy-c", "security", "Production credentials must never be stored in source control.", source="security"),
        Chunk("c4", "policy-d", "remote", "Remote work is permitted up to three days each week.", source="handbook"),
    ]


def test_bm25_finds_relevant_chunk():
    result = BM25Index(corpus()).search("How many annual leave days do employees receive?", top_k=2)
    assert result[0].chunk.chunk_id == "c1"


def test_rrf_is_deterministic():
    a, b = corpus()[:2]
    left = [Evidence(a, 1.0), Evidence(b, 0.5)]
    right = [Evidence(b, 1.0), Evidence(a, 0.5)]
    first = reciprocal_rank_fusion(left, right)
    second = reciprocal_rank_fusion(left, right)
    assert [(x.chunk.chunk_id, x.score) for x in first] == [(x.chunk.chunk_id, x.score) for x in second]


def test_reranker_prefers_matching_candidate():
    candidates = [Evidence(corpus()[1], 1.0), Evidence(corpus()[0], 0.9)]
    out = rerank("annual leave days", candidates)
    assert out[0].chunk.chunk_id == "c1"


def test_end_to_end_answer_is_grounded():
    response = RetrievalPipeline(corpus()).answer("What is required before booking travel expenses?")
    assert response.state is ResponseState.ANSWER
    assert response.evidence[0].chunk.chunk_id == "c2"
    assert "manager approval" in response.answer
