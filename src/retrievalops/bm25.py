from __future__ import annotations

from collections import Counter
import math

from .models import Chunk, Evidence
from .tokenize import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        self.chunks = tuple(chunks)
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(chunk.text) for chunk in self.chunks]
        self.lengths = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        self.term_freqs = [Counter(doc) for doc in self.docs]
        doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            doc_freq.update(set(doc))
        n = len(self.docs)
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, query: str, *, top_k: int = 20) -> list[Evidence]:
        qterms = tokenize(query)
        scored: list[Evidence] = []
        for chunk, tf, dl in zip(self.chunks, self.term_freqs, self.lengths):
            score = 0.0
            for term in qterms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(Evidence(chunk=chunk, score=score))
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:top_k]
