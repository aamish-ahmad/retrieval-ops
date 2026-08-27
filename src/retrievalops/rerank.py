from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Evidence


def rerank(query: str, candidates: list[Evidence], *, top_k: int = 8) -> list[Evidence]:
    """Candidate-local lexical reranker using character n-gram TF-IDF cosine similarity."""
    if not candidates:
        return []
    texts = [query, *(item.chunk.text for item in candidates)]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    rescored = [Evidence(item.chunk, float(score)) for item, score in zip(candidates, scores)]
    rescored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return rescored[:top_k]
