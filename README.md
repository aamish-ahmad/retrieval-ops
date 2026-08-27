# RetrievalOps

**Evidence-grounded RAG with hybrid retrieval, provenance, and safe answer control.**

RetrievalOps is a modular RAG reliability system for knowledge-heavy workflows where an answer is only useful if the supporting evidence can be inspected, traced, and rejected when it is not good enough.

## Why this exists

Typical RAG demos optimize for producing an answer. RetrievalOps treats retrieval quality, evidence admissibility, provenance, and safe failure behavior as first-class parts of the system.

Instead of returning a plausible response at any cost, the pipeline can:

- retrieve with independent sparse and dense signals;
- fuse and rerank candidate evidence;
- preserve document, section, chunk, and source provenance;
- reject inapplicable or stale evidence when version metadata is available;
- retry once when evidence is insufficient;
- clarify when evidence is ambiguous; and
- abstain rather than emit unsupported answer fields.

## Architecture

```text
documents + metadata
        ↓
normalized chunks + provenance
        ↓
BM25 sparse retrieval ───────────┐
                                 ├── reciprocal-rank fusion
LSA dense retrieval ─────────────┘
        ↓
candidate-local reranking
        ↓
evidence selection + admissibility
        ↓
grounded generation
        ↓
answer + provenance + response state
        ↓
ANSWER | CLARIFY | RETRY | ABSTAIN
```

## Core components

| Layer | Implementation | Purpose |
| --- | --- | --- |
| Sparse retrieval | Okapi BM25 | exact-term and lexical matching |
| Dense retrieval | TF-IDF + TruncatedSVD (LSA) | complementary semantic signal without an embedding API |
| Fusion | reciprocal-rank fusion | combine independent ranked lists |
| Reranking | character n-gram TF-IDF cosine | reorder the fused candidate set |
| Evidence control | admissibility + bounded retry/clarify/abstain states | prevent unsupported downstream answers |
| Generation | evidence-grounded extractive answer construction | keep answers tied to selected evidence |
| Traceability | document/section/chunk/source references | make support inspectable end to end |

## Reliability contract

RetrievalOps separates **retrieval relevance** from **evidence admissibility**.

A retrieved passage is not automatically allowed to support an answer. The evidence layer evaluates candidate state before generation.

Implemented response states:

- `ANSWER` — selected evidence is admissible;
- `CLARIFY` — the current evidence is ambiguous;
- `RETRY` — one bounded retry is permitted;
- `ABSTAIN` — support remains insufficient or inadmissible.

## Repository map

```text
src/retrievalops/
  bm25.py          sparse retrieval
  dense.py         LSA dense retrieval
  fusion.py        reciprocal-rank fusion
  rerank.py        candidate reranking
  evidence.py      admissibility + safe response states
  generation.py    grounded answer construction
  pipeline.py      end-to-end orchestration

tests/             deterministic unit and integration tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows CMD: .venv\Scripts\activate
python -m pip install -e ".[test]"
pytest
```

Minimal usage:

```python
from retrievalops import Chunk, RetrievalPipeline

chunks = [
    Chunk("1", "handbook", "travel", "Travel expenses require manager approval before booking."),
    Chunk("2", "handbook", "leave", "Employees receive twenty days of annual leave."),
]

pipeline = RetrievalPipeline(chunks)
response = pipeline.answer("What is required before booking travel expenses?")

print(response.state)
print(response.answer)
print(response.evidence[0].chunk.document)
```

## What this project demonstrates

- end-to-end RAG pipeline design rather than a single vector-search demo;
- hybrid sparse+dense retrieval and rank fusion;
- explicit contracts between retrieval, reranking, evidence, and generation stages;
- deterministic ranking and reproducible behavior;
- provenance-preserving evidence handling;
- evidence-to-answer traceability;
- safe failure states for weak or ambiguous support;
- focused tests around retrieval and admissibility behavior.

## Current scope

This version intentionally favors transparent, inspectable components over model-heavy infrastructure.

Current limitations:

- dense retrieval uses LSA rather than a neural embedding model;
- reranking is lexical rather than a learned cross-encoder;
- generation is extractive rather than abstractive;
- abstention is rule-based rather than probabilistically calibrated;
- production serving, background ingestion, persistence, deployment, and observability are outside this version.

Those are extension points, not capabilities claimed as already implemented.

## Design principle

> **A retrieved passage is not evidence until the system can justify using it.**
