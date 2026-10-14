"""Tests for BaseReranker: scoring, sorting, time decay, unknown reranker."""

from datetime import datetime

import numpy as np
import pytest

from zotero_arxiv_daily.protocol import CorpusPaper
from zotero_arxiv_daily.reranker.base import BaseReranker, get_reranker_cls
from zotero_arxiv_daily.sync.content import build_embedding_text
from tests.canned_responses import make_sample_paper, make_sample_corpus


class StubReranker(BaseReranker):
    """Reranker with a controlled similarity matrix for deterministic tests."""

    def __init__(self, sim_matrix: np.ndarray):
        self.config = None
        self._sim = sim_matrix

    def embed_texts(self, texts):
        raise AssertionError("controlled-stub reranker must not embed")

    def embedding_model_name(self):
        return "stub-model"

    def get_similarity_score(self, s1, s2):
        return self._sim


def test_rerank_scores_and_sorts():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(2)]

    # Paper 1 has higher similarity to all corpus papers
    sim = np.array([
        [0.1, 0.1, 0.1],  # paper 0 — low
        [0.9, 0.9, 0.9],  # paper 1 — high
    ])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "Paper 1"
    assert ranked[1].title == "Paper 0"
    assert ranked[0].score > ranked[1].score


def test_rerank_time_decay_weighting():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title="P")]

    # Only similar to the oldest paper (index 2 after reverse-sort by date)
    sim = np.array([[0.0, 0.0, 1.0]])
    reranker = StubReranker(sim)
    ranked_old = reranker.rerank(papers, corpus)
    score_old = ranked_old[0].score

    # Only similar to the newest paper (index 0 after reverse-sort by date)
    papers2 = [make_sample_paper(title="P")]
    sim2 = np.array([[1.0, 0.0, 0.0]])
    reranker2 = StubReranker(sim2)
    ranked_new = reranker2.rerank(papers2, corpus)
    score_new = ranked_new[0].score

    # Newest corpus paper gets higher time-decay weight, so score should be higher
    assert score_new > score_old


def test_rerank_single_candidate_single_corpus():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper()]
    sim = np.array([[0.5]])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")


class FixedVectorReranker(BaseReranker):
    """embed_texts returns a deterministic per-text vector."""

    def __init__(self, vectors: dict):
        self.config = None
        self._vectors = vectors
        self.embedded: list[list[str]] = []

    def embed_texts(self, texts):
        self.embedded.append(list(texts))
        return np.stack([self._vectors[t] for t in texts]).astype(np.float32)

    def embedding_model_name(self):
        return "fixed-model"


def test_rerank_corpus_matrix_reordered_with_recency_sort():
    """C1 regression: cached matrix rows must follow the date sort, or the
    time-decay weights attach to the wrong papers."""
    old = CorpusPaper(
        title="Old", abstract="old body",
        added_date=datetime(2020, 1, 1), paths=[], key="OLD",
    )
    new = CorpusPaper(
        title="New", abstract="new body",
        added_date=datetime(2026, 1, 1), paths=[], key="NEW",
    )
    candidate = make_sample_paper(title="Q", abstract="question")
    t_old = build_embedding_text(old.title, old.abstract)
    t_new = build_embedding_text(new.title, new.abstract)
    t_q = build_embedding_text(candidate.title, candidate.abstract)
    # orthogonal, already-normalised 2-D vectors
    v_old = np.array([1.0, 0.0], dtype=np.float32)
    v_new = np.array([0.0, 1.0], dtype=np.float32)
    v_q = np.array([0.6, 0.8], dtype=np.float32)  # leans "new"
    vectors = {t_old: v_old, t_new: v_new, t_q: v_q}

    # Cached path: corpus supplied in STORAGE order [old, new], matrix aligned
    # to that order; rerank must reorder to [new, old] internally.
    reranker_cache = FixedVectorReranker(vectors)
    cached = reranker_cache.rerank(
        [candidate], [old, new], np.stack([v_old, v_new])
    )

    # Reference path: no matrix, both sides embedded after the same sort.
    reranker_ref = FixedVectorReranker(vectors)
    reference = reranker_ref.rerank(
        [make_sample_paper(title="Q", abstract="question")], [old, new]
    )

    assert np.isclose(cached[0].score, reference[0].score)
    # candidate is more similar to the newer paper, so the score must reflect
    # the newest-paper weight rather than the oldest-paper weight
    assert cached[0].score > 0


def test_rerank_corpus_matrix_asserts_row_count_mismatch():
    corpus = make_sample_corpus(2)
    papers = [make_sample_paper(title="P")]
    q_text = build_embedding_text(papers[0].title, papers[0].abstract)
    reranker = FixedVectorReranker({q_text: np.zeros(4, dtype=np.float32)})
    with pytest.raises(AssertionError, match="corpus_embeddings rows"):
        reranker.rerank(papers, corpus, np.zeros((3, 4), dtype=np.float32))
