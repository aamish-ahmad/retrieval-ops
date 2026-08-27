"""Hybrid sparse+dense retrieval, reciprocal-rank fusion, and local reranking."""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .schema import Chunk, Evidence

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold()))


class HybridRetriever:
    """Deterministic BM25 + LSA retrieval with RRF and candidate-local reranking."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        sparse_top_k: int = 20,
        dense_top_k: int = 20,
        fused_top_k: int = 20,
        reranked_top_k: int = 8,
        rrf_constant: int = 60,
        dense_dimensions: int = 128,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        self.chunks = tuple(chunks)
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k
        self.fused_top_k = fused_top_k
        self.reranked_top_k = reranked_top_k
        self.rrf_constant = rrf_constant

        self._tokenized = [_tokens(chunk.text) for chunk in self.chunks]
        self._lengths = np.array([max(1, len(tokens)) for tokens in self._tokenized], dtype=float)
        self._average_length = float(self._lengths.mean())
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokenized:
            self._document_frequency.update(set(tokens))

        self._dense_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        matrix = self._dense_vectorizer.fit_transform(chunk.text for chunk in self.chunks)
        max_components = min(matrix.shape) - 1
        self._svd: TruncatedSVD | None = None
        self._dense_vectors: np.ndarray | None = None
        if max_components >= 1:
            self._svd = TruncatedSVD(
                n_components=min(dense_dimensions, max_components), n_iter=7, random_state=0
            )
            self._dense_vectors = self._normalise(self._svd.fit_transform(matrix))

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)

    def _bm25(self, question: str) -> np.ndarray:
        query_tokens = _tokens(question)
        scores = np.zeros(len(self.chunks))
        for index, tokens in enumerate(self._tokenized):
            counts = Counter(tokens)
            denominator_length = self._lengths[index] / self._average_length
            for token in query_tokens:
                frequency = counts[token]
                if not frequency:
                    continue
                idf = math.log(
                    1
                    + (len(self.chunks) - self._document_frequency[token] + 0.5)
                    / (self._document_frequency[token] + 0.5)
                )
                scores[index] += idf * (frequency * 2.2) / (
                    frequency + 1.2 * (1 - 0.75 + 0.75 * denominator_length)
                )
        return scores

    def _dense(self, question: str) -> np.ndarray:
        if self._svd is None or self._dense_vectors is None:
            return np.zeros(len(self.chunks))
        query = self._svd.transform(self._dense_vectorizer.transform([question]))
        normalised_query = self._normalise(query)[0]
        return self._dense_vectors @ normalised_query

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
        positive = [index for index, score in enumerate(scores) if score > 0]
        return sorted(positive, key=lambda index: (-scores[index], index))[:limit]

    def retrieve(self, question: str) -> list[Evidence]:
        """Return top evidence after independent retrieval, RRF, and local reranking."""
        sparse_scores = self._bm25(question)
        dense_scores = self._dense(question)
        sparse_ranking = self._top_indices(sparse_scores, self.sparse_top_k)
        dense_ranking = self._top_indices(dense_scores, self.dense_top_k)
        fusion_scores: Counter[int] = Counter()
        for ranking in (sparse_ranking, dense_ranking):
            for position, index in enumerate(ranking, 1):
                fusion_scores[index] += 1 / (self.rrf_constant + position)
        candidate_indices = sorted(
            fusion_scores, key=lambda index: (-fusion_scores[index], index)
        )[: self.fused_top_k]
        if not candidate_indices:
            return []
        candidate_texts = [self.chunks[index].text for index in candidate_indices]
        reranker = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        candidate_matrix = reranker.fit_transform(candidate_texts)
        query_vector = reranker.transform([question])
        rerank_scores = (candidate_matrix @ query_vector.T).toarray().ravel()
        ordered = sorted(
            range(len(candidate_indices)),
            key=lambda position: (-rerank_scores[position], candidate_indices[position]),
        )[: self.reranked_top_k]
        return [
            Evidence(
                chunk=self.chunks[candidate_indices[position]],
                sparse_score=float(sparse_scores[candidate_indices[position]]),
                dense_score=float(dense_scores[candidate_indices[position]]),
                fusion_score=float(fusion_scores[candidate_indices[position]]),
                rerank_score=float(rerank_scores[position]),
            )
            for position in ordered
        ]
