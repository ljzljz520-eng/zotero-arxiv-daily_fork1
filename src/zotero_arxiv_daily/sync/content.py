"""Deterministic text normalization and content-addressed cache keys.

The exact same functions MUST be used for Zotero corpus items and for daily
candidate papers, otherwise the embedding spaces would be fed differently
composed inputs and cached vectors could not be compared with fresh ones.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")

# Fixed separator between normalized title and abstract in the embedding input.
EMBEDDING_SEPARATOR = "\n"


def normalize_text(value: object) -> str:
    """NFKC-normalize, collapse all whitespace runs to a single space, strip.

    Non-strings (including None) are coerced to str; None becomes "".
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFKC", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def build_embedding_text(title: object, abstract: object) -> str:
    """Canonical embedding input for a paper.

    Normalized title and abstract joined by :data:`EMBEDDING_SEPARATOR`.
    A missing/empty side yields a deterministic leading/trailing separator so
    that ("", "x") and ("x", "") never collide.
    """
    return normalize_text(title) + EMBEDDING_SEPARATOR + normalize_text(abstract)


def content_hash(title: object, abstract: object) -> str:
    """sha256 hex digest of the canonical embedding text."""
    text = build_embedding_text(title, abstract)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
