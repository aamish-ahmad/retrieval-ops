# Reproducibility

## Corpus contract

Provide one JSON object per line with these required fields:

```json
{"chunk_id":"policy-1","document_id":"travel-policy","section":"Approval","text":"Emergency travel requires director approval.","version":"2026-01","effective_state":"active"}
```

`version` and `effective_state` are optional in the file format. When a version is requested at query time, only an exactly matching version is eligible. The default eligible lifecycle states are `active` and `current`.

## Deterministic settings

The default configuration is checked in at `configs/retrievalops.yml`. The LSA stage fixes its random seed at zero; retrieval and tie handling are ordered deterministically by score and chunk position.

## Local inspection

Install with `pip install -e '.[dev]'`, run `pytest -q`, then use the `retrievalops` command shown in the README. The included example corpus is synthetic and exists solely to make the end-to-end behavior inspectable without private inputs.
