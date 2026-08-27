from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Return deterministic NFKC-normalized, case-folded word tokens."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_WORD_RE.findall(normalized))
