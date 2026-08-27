# Architecture

RetrievalOps implements a deliberately inspectable RAG path. The pipeline accepts only a validated local corpus and returns a response together with the evidence that determined it.

## Component boundaries

| Component | Responsibility | Output |
| --- | --- | --- |
| `ingest` | Validate JSONL records and preserve chunk metadata | `Chunk` records |
| `retrieval` | BM25 and LSA retrieval, reciprocal-rank fusion, local reranking | ranked `Evidence` |
| `reliability` | Enforce lifecycle, version, ambiguity, and score conditions | response state + selected evidence |
| `generation` | Construct only an evidence-derived answer and claim trace | `Response` |
| `pipeline` | Compose the stages without merging their responsibilities | query interface |

## Data flow

1. A JSONL loader validates required identifiers, section labels, text, and unique chunk IDs.
2. BM25 scores lexical overlap; TF-IDF plus truncated SVD provides a deterministic LSA signal.
3. Reciprocal-rank fusion joins the independent rankings. A character n-gram TF-IDF reranker evaluates only the fused candidate set.
4. Evidence selection rejects inactive or wrong-version chunks. It distinguishes insufficient support, document ambiguity, and a bounded repeat attempt.
5. Only `ANSWER` reaches response construction. The selected chunk text becomes the claim and is paired with its source identifiers.

The trace is intentionally simple: every generated claim names the exact chunk, document, and section from which it was constructed. This makes evidence inspection possible without interpreting hidden model state.
