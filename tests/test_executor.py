"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, incremental-sync E2E."""

import hashlib
import smtplib
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf, open_dict
from pyzotero.errors import UserNotAuthorisedError

from zotero_arxiv_daily.executor import Executor, normalize_path_patterns
from zotero_arxiv_daily.protocol import CorpusPaper
from zotero_arxiv_daily.sync.zotero_client import SyncPermissionError

_ZOTERO_PATCH_TARGET = "zotero_arxiv_daily.sync.zotero_client.zotero.Zotero"


# ---------------------------------------------------------------------------
# normalize_path_patterns — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


# ---------------------------------------------------------------------------
# filter_corpus — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


# ---------------------------------------------------------------------------
# E2E: Executor.run() over the incremental sync layer
# ---------------------------------------------------------------------------


def _stub_arxiv(monkeypatch, retrieved):
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: retrieved,
    )
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)


def _recording_openai_client(monkeypatch):
    """Canned OpenAI whose embedding inputs are recorded."""
    from tests.canned_responses import make_stub_openai_client

    stub_client = make_stub_openai_client()
    recorded: list[str] = []
    original_create = stub_client.embeddings.create

    def create(**kwargs):
        recorded.extend(kwargs.get("input", []))
        return original_create(**kwargs)

    stub_client.embeddings.create = create
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    return stub_client, recorded


def test_run_two_rounds_reuses_corpus_embeddings(config, monkeypatch):
    """TR-5.1: round 1 builds the cache; round 2 embeds candidates only."""
    from tests.canned_responses import (
        make_sample_paper,
        make_stub_zotero_client,
    )

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr(_ZOTERO_PATCH_TARGET, lambda *a, **kw: stub_zot)

    _stub_arxiv(
        monkeypatch,
        [make_sample_paper(title="E2E Paper 1"),
         make_sample_paper(title="E2E Paper 2")],
    )
    sent: list = []
    monkeypatch.setattr(smtplib.SMTP, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(smtplib.SMTP, "sendmail",
                        lambda self, *a: sent.append(a))
    monkeypatch.setattr(smtplib.SMTP, "starttls", lambda self: None)
    monkeypatch.setattr(smtplib.SMTP, "login", lambda self, *a: None)
    monkeypatch.setattr(smtplib.SMTP, "quit", lambda self: None)

    # ---- round 1: full build ----
    _, recorded_1 = _recording_openai_client(monkeypatch)
    Executor(config).run()

    assert any("Stub Paper 1" in t for t in recorded_1)
    assert any("Stub Paper 2" in t for t in recorded_1)
    assert any("E2E Paper 1" in t for t in recorded_1)

    # ---- round 2: incremental, corpus cache hit ----
    _, recorded_2 = _recording_openai_client(monkeypatch)
    Executor(config).run()

    assert not any("Stub Paper" in t for t in recorded_2), (
        "corpus texts must not be re-embedded on the second run"
    )
    assert any("E2E Paper 1" in t for t in recorded_2)
    assert any("E2E Paper 2" in t for t in recorded_2)

    assert len(sent) == 2
    for _, _, email_body in sent:
        assert "text/html" in email_body


def test_filtered_corpus_vectors_align_by_key(config, monkeypatch):
    """TR-5.2: after glob filtering, matrix rows follow paper keys/order."""
    from tests.canned_responses import (
        make_sample_paper,
        make_stub_zotero_client,
    )

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        # "survey" (ITEM2's path) is excluded; "survey/topic-a" (ITEM1) stays
        config.zotero.ignore_path = ["survey"]

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr(_ZOTERO_PATCH_TARGET, lambda *a, **kw: stub_zot)
    _stub_arxiv(monkeypatch, [make_sample_paper(title="Candidate X")])

    # Deterministic, text-distinct embeddings so row alignment is checkable.
    from tests.canned_responses import make_stub_openai_client

    stub_client = make_stub_openai_client()

    def create(**kwargs):
        data = []
        for i, text in enumerate(kwargs.get("input", [])):
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            vec = np.random.RandomState(seed).rand(3).astype(np.float32).tolist()
            data.append(SimpleNamespace(embedding=vec, index=i, object="embedding"))
        return SimpleNamespace(data=data, model="m", object="list")

    stub_client.embeddings.create = create
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    sent: list = []
    monkeypatch.setattr(smtplib.SMTP, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(smtplib.SMTP, "sendmail",
                        lambda self, *a: sent.append(a))
    monkeypatch.setattr(smtplib.SMTP, "starttls", lambda self: None)
    monkeypatch.setattr(smtplib.SMTP, "login", lambda self, *a: None)
    monkeypatch.setattr(smtplib.SMTP, "quit", lambda self: None)

    executor = Executor(config)
    captured = {}

    def spy(candidates, corpus, corpus_embeddings=None):
        captured["corpus"] = corpus
        captured["matrix"] = corpus_embeddings
        return executor.reranker.__class__.rerank(
            executor.reranker, candidates, corpus, corpus_embeddings
        )

    executor.reranker.rerank = spy
    executor.run()

    rerank_corpus = captured["corpus"]
    matrix = captured["matrix"]
    assert [p.key for p in rerank_corpus] == ["ITEM1"]
    assert matrix.shape == (1, 3)

    full_papers, full_matrix = executor.syncer.load_corpus()
    full_index = {p.key: i for i, p in enumerate(full_papers)}
    np.testing.assert_allclose(
        matrix[0], full_matrix[full_index["ITEM1"]]
    )
    assert len(sent) == 1


def test_sync_permission_error_aborts_without_email(config, monkeypatch):
    """TR-5.3: permission failure propagates; no email is sent."""
    from tests.canned_responses import (
        make_sample_paper,
        make_stub_openai_client,
        make_stub_zotero_client,
    )

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"

    stub_zot = make_stub_zotero_client(
        key_info=UserNotAuthorisedError("403 revoked mid-run")
    )
    monkeypatch.setattr(_ZOTERO_PATCH_TARGET, lambda *a, **kw: stub_zot)
    _stub_arxiv(monkeypatch, [make_sample_paper(title="Never Reached")])

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    sent: list = []
    monkeypatch.setattr(smtplib.SMTP, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(smtplib.SMTP, "sendmail",
                        lambda self, *a: sent.append(a))
    monkeypatch.setattr(smtplib.SMTP, "starttls", lambda self: None)
    monkeypatch.setattr(smtplib.SMTP, "login", lambda self, *a: None)
    monkeypatch.setattr(smtplib.SMTP, "quit", lambda self: None)

    with pytest.raises(SyncPermissionError):
        Executor(config).run()
    assert sent == []


def test_run_no_papers_send_empty_false(config, monkeypatch):
    """When no papers are found and send_empty=false, no email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr(_ZOTERO_PATCH_TARGET, lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0, "No email should be sent when no papers and send_empty=false"


def test_run_no_papers_send_empty_true(config, monkeypatch):
    """When no papers are found and send_empty=true, empty email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr(_ZOTERO_PATCH_TARGET, lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Email should be sent even with no papers when send_empty=true"
    _, _, body = sent[0]
    assert "text/html" in body
