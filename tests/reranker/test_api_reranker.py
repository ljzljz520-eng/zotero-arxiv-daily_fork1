"""Tests for ApiReranker — uses stub OpenAI client via monkeypatch."""

from types import SimpleNamespace

import numpy as np
from omegaconf import open_dict

from zotero_arxiv_daily.reranker.api import ApiReranker
from tests.canned_responses import make_sample_corpus, make_sample_paper


def test_api_reranker_similarity_shape(config, patch_openai):
    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


def test_api_reranker_batching(config, patch_openai):
    reranker = ApiReranker(config)
    s1 = [f"text {i}" for i in range(5)]
    s2 = [f"corpus {i}" for i in range(3)]
    score = reranker.get_similarity_score(s1, s2)
    assert score.shape == (5, 3)


class _RecordingEmbeddings:
    def __init__(self, dim=3, start=0):
        self.dim = dim
        self.calls = []
        self._position = start

    def create(self, **kwargs):
        batch = list(kwargs["input"])
        self.calls.append(batch)
        items = [
            SimpleNamespace(embedding=[float(self._position + i), 0.0, 0.0],
                            index=i, object="embedding")
            for i in range(len(batch))
        ]
        self._position += len(batch)
        # deliberately reverse data order; the response index must win
        return SimpleNamespace(
            data=items[::-1], model=kwargs["model"], object="list"
        )


def _recording_client(start=0):
    emb = _RecordingEmbeddings(start=start)
    return SimpleNamespace(embeddings=emb), emb


def test_embed_texts_batches_and_orders_by_index(config, monkeypatch):
    client, emb = _recording_client()
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI",
                        lambda **kwargs: client)
    with open_dict(config):
        config.reranker.api.batch_size = 2
    reranker = ApiReranker(config)
    mat = reranker.embed_texts(["a", "b", "c", "d", "e"])
    assert mat.shape == (5, 3)
    assert len(emb.calls) == 3  # 2 + 2 + 1
    assert emb.calls[0] == ["a", "b"]
    assert emb.calls[-1] == ["e"]
    # row i corresponds to input i despite reversed response data
    assert mat[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_embed_texts_reuses_single_client(config, monkeypatch):
    client, emb = _recording_client()
    built = []
    monkeypatch.setattr(
        "zotero_arxiv_daily.reranker.api.OpenAI",
        lambda **kwargs: (built.append(1), client)[1],
    )
    reranker = ApiReranker(config)
    assert reranker._client is None
    reranker.embed_texts(["x"])
    reranker.embed_texts(["y"])
    assert len(built) == 1
    assert reranker._client is client


def test_rerank_with_corpus_embeddings_only_embeds_candidates(config, monkeypatch):
    # start=1 keeps candidate vectors non-zero (avoids NaN normalization)
    client, emb = _recording_client(start=1)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI",
                        lambda **kwargs: client)
    reranker = ApiReranker(config)

    papers = [make_sample_paper(title="Cand A"), make_sample_paper(title="Cand B")]
    corpus = make_sample_corpus(3)
    corpus_matrix = np.eye(3, dtype=np.float32)

    ranked = reranker.rerank(papers, corpus, corpus_embeddings=corpus_matrix)

    assert len(ranked) == 2
    # only candidate texts were sent to the embedding endpoint
    all_inputs = [t for batch in emb.calls for t in batch]
    assert len(all_inputs) == 2
    assert all(t.startswith("Cand ") for t in all_inputs)
    assert not any("Corpus Paper" in t for t in all_inputs)


def test_rerank_rejects_misaligned_corpus_matrix(config, monkeypatch):
    client, _ = _recording_client()
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI",
                        lambda **kwargs: client)
    reranker = ApiReranker(config)
    with np.testing.assert_raises(AssertionError):
        reranker.rerank(
            [make_sample_paper()],
            make_sample_corpus(3),
            corpus_embeddings=np.eye(2, dtype=np.float32),
        )


def test_embedding_fingerprint_base_and_dimension(config):
    reranker = ApiReranker(config)
    base = reranker.embedding_fingerprint_base()
    assert base == "api:text-embedding-3-large"
    fp = reranker.resolve_fingerprint(base, 3072)
    assert fp == "api:text-embedding-3-large:dim3072"
    assert reranker.split_fingerprint(fp) == ("api:text-embedding-3-large", 3072)
    assert reranker.split_fingerprint(base) == ("api:text-embedding-3-large", None)


def test_embedding_fingerprint_changes_with_explicit_version(config):
    reranker = ApiReranker(config)
    with open_dict(config):
        config.reranker.embedding_model_version = "2026-10-v2"
    base = reranker.embedding_fingerprint_base()
    assert base == "api:text-embedding-3-large:v2026-10-v2"
