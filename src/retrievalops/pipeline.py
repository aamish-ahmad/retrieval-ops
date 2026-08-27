from __future__ import annotations

from .bm25 import BM25Index
from .dense import LSAIndex
from .evidence import select_evidence
from .fusion import reciprocal_rank_fusion
from .generation import generate_extractive
from .models import Chunk, Response
from .rerank import rerank


class RetrievalPipeline:
    """Small end-to-end hybrid retrieval pipeline with explicit evidence control."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.bm25 = BM25Index(chunks)
        self.dense = LSAIndex(chunks)

    def answer(self, query: str, *, target_version: str | None = None) -> Response:
        sparse = self.bm25.search(query, top_k=20)
        dense = self.dense.search(query, top_k=20)
        fused = reciprocal_rank_fusion(sparse, dense, top_k=20)
        ranked = rerank(query, fused, top_k=8)
        state, selected, reason = select_evidence(ranked, target_version=target_version)
        return generate_extractive(state, selected, reason)
