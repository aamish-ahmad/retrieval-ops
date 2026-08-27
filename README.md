# RetrievalOps

RetrievalOps is an evidence-grounded RAG reliability system for teams that need answers they can inspect, trace, and safely withhold when the corpus does not support a conclusion.

## The problem

Knowledge systems often return fluent text without making its source, freshness, or scope clear. That creates operational risk: a user cannot tell whether an answer is supported by an approved document, a superseded policy, or no source at all.

RetrievalOps turns a local document corpus into a controlled answer path. It retrieves evidence, ranks it with multiple signals, checks whether it is usable, and produces either a source-grounded answer or an explicit safe state.

## Architecture

```text
JSONL corpus
    -> BM25 sparse retrieval + LSA dense retrieval
    -> reciprocal-rank fusion
    -> candidate-local character n-gram reranking
    -> evidence selection and lifecycle/version checks
    -> grounded extractive response + claim trace
    -> ANSWER | CLARIFY | RETRY | ABSTAIN
```

The system keeps retrieval, evidence acceptance, and answer construction separate. Each answer carries ranked evidence and a claim trace with the chunk, document, and section that support the displayed claim.

## Reliability behavior

- Hybrid retrieval combines BM25 lexical matching with deterministic LSA semantic matching.
- Reciprocal-rank fusion reduces dependence on a single retrieval signal.
- A local reranker narrows fused candidates to the most question-relevant evidence.
- Evidence is filtered by lifecycle state and optional version before an answer is permitted.
- Weak or absent evidence returns `RETRY`, then `ABSTAIN` on a bounded repeat attempt.
- Equally ranked evidence from different documents returns `CLARIFY` instead of asserting a choice.
- `ANSWER` is extractive: the response text is copied from selected evidence, with provenance attached.

## Stack

- Python 3.11+
- scikit-learn for TF-IDF, truncated SVD, and vector operations
- pytest for deterministic behavior checks
- JSONL for portable, inspectable local corpora

## Run locally

Create an environment, install the package with its test tools, then run the test suite:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Windows CMD activation:

```cmd
.venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

Inspect the included synthetic corpus:

```bash
retrievalops --corpus examples/demo_corpus.jsonl --question "Who may approve emergency travel?"
```

The CLI prints the response state, answer text when allowed, selected source IDs, and the claim trace. It is deliberately local and uses only the supplied corpus.

## Repository guide

- `src/retrievalops/` — ingest, retrieval, reliability checks, grounded response construction, and CLI
- `tests/` — retrieval and safe-behavior tests
- `configs/` — documented retrieval and evidence thresholds
- `docs/ARCHITECTURE.md` — component boundaries and data flow
- `docs/REPRODUCIBILITY.md` — corpus contract and deterministic run notes
- `examples/` — a synthetic corpus safe to share publicly

## Engineering capabilities demonstrated

This project demonstrates modular RAG system design, deterministic ranking pipelines, provenance-aware data models, explicit failure states, version and lifecycle filtering, reproducible local execution, and behavior-focused testing.

## Current limitations and extensions

The current dense stage is LSA rather than an embedding model, reranking is lexical, and the answer constructor is extractive rather than generative. The repository does not include a service layer, persistent index, access controls, automated evaluation harness, or a user interface. Natural future extensions include benchmarked embedding and cross-encoder stages, richer claim segmentation, document-level permissions, observability, and a separately evaluated service interface.
