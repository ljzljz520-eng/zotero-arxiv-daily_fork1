"""Tests for text normalization and content-addressed hashing (TR-2.1)."""

import unicodedata

from zotero_arxiv_daily.sync.content import (
    build_embedding_text,
    content_hash,
    normalize_text,
)


def test_normalize_collapses_all_whitespace_and_strips():
    assert normalize_text("  a   b\tc\nd  ") == "a b c d"


def test_normalize_nfkc_full_width_and_nonbreaking_space():
    # full-width ASCII + ideographic space
    raw = "Ｈｅｌｌｏ　World"
    assert normalize_text(raw) == "Hello World"
    # NFKC on ligatures
    assert normalize_text(unicodedata.normalize("NFKC", "ﬁle")) == "file"


def test_normalize_none_and_non_string():
    assert normalize_text(None) == ""
    assert normalize_text(123) == "123"


def test_case_and_unicode_letters_preserved():
    assert normalize_text("BERT 中文") == "BERT 中文"


def test_build_embedding_text_deterministic_and_separator_safe():
    t1 = build_embedding_text(" Title ", " Abs ")
    t2 = build_embedding_text("Title", "Abs")
    assert t1 == t2 == "Title\nAbs"
    # missing sides must not collide
    assert build_embedding_text("", "x") == "\nx"
    assert build_embedding_text("x", "") == "x\n"
    assert build_embedding_text(None, None) == "\n"


def test_content_hash_stable_and_sensitive():
    assert content_hash("T", "A") == content_hash("  T ", "A")
    assert content_hash("T", "A") == content_hash("T", "  A  ")
    assert content_hash("T", "A") != content_hash("T", "B")
    assert content_hash("T", "A") != content_hash("T2", "A")
    # whitespace variants (tabs/newlines/multiple spaces) hash the same
    assert content_hash("T\tX", "A\nB") == content_hash("T X", "A B")
    assert len(content_hash("T", "A")) == 64  # sha256 hex
