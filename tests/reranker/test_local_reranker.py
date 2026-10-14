"""Tests for LocalReranker.

The real SentenceTransformer integration is marked ``slow``; offline tests
inject a fake encoder to verify batching/reuse/fingerprint logic.
"""

import hashlib
import json

import numpy as np
import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.reranker.local import LocalReranker


class FakeEncoder:
    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.arange(len(texts) * self.dim, dtype=np.float32).reshape(
            len(texts), self.dim
        )


def test_local_embed_texts_with_fake_encoder(config):
    reranker = LocalReranker(config)
    fake = FakeEncoder()
    reranker._encoder = fake  # bypass model download / lazy loader
    mat = reranker.embed_texts(["a", "b"])
    assert mat.shape == (2, 4)
    assert mat.dtype == np.float32
    reranker.embed_texts(["c"])
    # encoder instance is reused across calls within one process
    assert len(fake.calls) == 2
    assert fake.calls[1][0] == ["c"]


def test_local_embed_texts_empty(config):
    reranker = LocalReranker(config)
    reranker._encoder = FakeEncoder()
    assert reranker.embed_texts([]).shape == (0, 0)


def test_local_fingerprint(config):
    reranker = LocalReranker(config)
    base = reranker.embedding_fingerprint_base()
    kwargs = dict(config.reranker.local.get("encode_kwargs") or {})
    if kwargs:
        canonical = json.dumps(
            kwargs, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        assert base == f"local:{config.reranker.local.model}:p{digest}"
    else:
        assert base == f"local:{config.reranker.local.model}"
    fp = reranker.resolve_fingerprint(base, 1024)
    assert fp.endswith(":dim1024")


def test_local_fingerprint_changes_with_encode_kwargs(config):
    with open_dict(config):
        config.reranker.local.encode_kwargs = {"task": "retrieval.passage"}
    base_a = LocalReranker(config).embedding_fingerprint_base()
    with open_dict(config):
        config.reranker.local.encode_kwargs = {"task": "classification"}
    base_b = LocalReranker(config).embedding_fingerprint_base()
    with open_dict(config):
        config.reranker.local.encode_kwargs = {}
    base_c = LocalReranker(config).embedding_fingerprint_base()
    assert base_a != base_b
    assert base_c == f"local:{config.reranker.local.model}"


@pytest.mark.slow
def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
