"""Command-line interface for local, inspectable RetrievalOps runs."""

from __future__ import annotations

import argparse
import json

from .ingest import load_jsonl
from .pipeline import EvidenceGroundedRAG


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an evidence-grounded local RAG query.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--retry-attempt", type=int, choices=(0, 1), default=0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    response = EvidenceGroundedRAG(load_jsonl(args.corpus)).answer(
        args.question,
        requested_version=args.version,
        retry_attempt=args.retry_attempt,
    )
    payload = {
        "state": response.state.value,
        "answer": response.answer,
        "reason": response.reason,
        "evidence": [
            {
                "chunk_id": item.chunk.chunk_id,
                "document_id": item.chunk.document_id,
                "section": item.chunk.section,
                "version": item.chunk.version,
                "rerank_score": item.rerank_score,
            }
            for item in response.evidence
        ],
        "claim_trace": list(response.claim_trace),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
