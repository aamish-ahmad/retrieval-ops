from __future__ import annotations

from collections import defaultdict

from .models import Evidence


def reciprocal_rank_fusion(*rankings: list[Evidence], k: int = 60, top_k: int = 20) -> list[Evidence]:
    """Fuse ranked lists with equal-weight reciprocal-rank fusion."""
    scores: dict[str, float] = defaultdict(float)
    chunks = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.chunk_id
            scores[cid] += 1.0 / (k + rank)
            chunks[cid] = item.chunk
    fused = [Evidence(chunks[cid], score) for cid, score in scores.items()]
    fused.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return fused[:top_k]
