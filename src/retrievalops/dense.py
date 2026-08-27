from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .models import Chunk, Evidence


class LSAIndex:
    """Corpus-fitted non-neural dense retriever using TF-IDF + truncated SVD."""

    def __init__(self, chunks: list[Chunk], *, n_components: int = 128, random_state: int = 0) -> None:
        if len(chunks) < 2:
            raise ValueError("LSAIndex needs at least two chunks")
        self.chunks = tuple(chunks)
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.chunks)
        max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
        if max_components < 1:
            raise ValueError("corpus has insufficient vocabulary for LSA")
        components = min(n_components, max_components)
        self.svd = TruncatedSVD(n_components=components, random_state=random_state)
        self.doc_vectors = normalize(self.svd.fit_transform(matrix))

    def search(self, query: str, *, top_k: int = 20) -> list[Evidence]:
        q = self.vectorizer.transform([query])
        qv = normalize(self.svd.transform(q))[0]
        scores = np.asarray(self.doc_vectors @ qv).ravel()
        order = np.argsort(-scores)
        results: list[Evidence] = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(Evidence(self.chunks[int(idx)], score))
            if len(results) >= top_k:
                break
        return results
