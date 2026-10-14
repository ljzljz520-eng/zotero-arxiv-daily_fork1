import hashlib
import json
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
from ..sync.content import build_embedding_text
import numpy as np
from typing import Type


class BaseReranker(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Embedding provider interface used by the incremental sync cache.
    # ------------------------------------------------------------------

    #: short backend id, overridden by concrete rerankers ("api"/"local")
    embedding_backend: str = "base"

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of raw texts.

        Returns a float32 matrix ``[len(texts), dim]``. Vectors are returned
        unnormalized; cosine normalization happens in this base class.
        """
        raise NotImplementedError

    @abstractmethod
    def embedding_model_name(self) -> str:
        """Configured model identifier (API model name or HF model id)."""
        raise NotImplementedError

    def embedding_model_version(self) -> str | None:
        """Optional explicit model/version pin for cache invalidation.

        Set ``reranker.embedding_model_version`` to force a fingerprint change
        when the configured model name stays the same but served weights
        change.
        """
        value = self.config.reranker.get("embedding_model_version", None)
        return value or None

    def embedding_model_params(self) -> dict:
        """Stable parameters that materially change the produced vectors.

        Subclasses should expose config-derived encode parameters here (encode
        task, prompt name, dimensionality overrides, ...). A canonical hash of
        them is folded into the fingerprint so that changing any such
        parameter invalidates the content-addressed embedding cache.
        """
        return {}

    def embedding_fingerprint_base(self) -> str:
        """Fingerprint without the runtime-discovered dimension suffix."""
        core = f"{self.embedding_backend}:{self.embedding_model_name()}"
        params = self.embedding_model_params()
        if params:
            canonical = json.dumps(
                params, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
            core += f":p{digest}"
        version = self.embedding_model_version()
        if version:
            core += f":v{version}"
        return core

    def embedding_batch_size(self) -> int:
        """How many texts the sync layer should send per embed_texts call."""
        return 64

    @classmethod
    def resolve_fingerprint(cls, base: str, dim: int) -> str:
        """Full fingerprint, appending the observed vector dimensionality."""
        return f"{base}:dim{int(dim)}"

    @staticmethod
    def split_fingerprint(fingerprint: str) -> tuple[str, int | None]:
        """Inverse of :meth:`resolve_fingerprint`: (base, dim or None)."""
        marker = ":dim"
        if marker in fingerprint:
            base, _, dim_str = fingerprint.rpartition(marker)
            if dim_str.isdigit():
                return base, int(dim_str)
        return fingerprint, None

    # ------------------------------------------------------------------
    # Similarity / reranking
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    @staticmethod
    def _cosine(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        return np.dot(BaseReranker._normalize(emb1), BaseReranker._normalize(emb2).T)

    def rerank(
        self,
        candidates: list[Paper],
        corpus: list[CorpusPaper],
        corpus_embeddings: np.ndarray | None = None,
    ) -> list[Paper]:
        # Sort corpus by recency; the embedding matrix (if given) MUST be
        # reordered with the same permutation or time-decay weights get
        # applied to the wrong papers.
        if corpus_embeddings is not None:
            assert corpus_embeddings.shape[0] == len(corpus), (
                f"corpus_embeddings rows {corpus_embeddings.shape[0]} "
                f"!= corpus size {len(corpus)}"
            )
        order = sorted(
            range(len(corpus)), key=lambda i: corpus[i].added_date, reverse=True
        )
        corpus = [corpus[i] for i in order]
        if corpus_embeddings is not None:
            corpus_embeddings = np.asarray(corpus_embeddings)[order]
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight: np.ndarray = time_decay_weight / time_decay_weight.sum()
        candidate_texts = [build_embedding_text(c.title, c.abstract) for c in candidates]
        if corpus_embeddings is not None:
            # Cache hit path: embed only the daily candidates.
            candidate_embeddings = self.embed_texts(candidate_texts)
            sim = self._cosine(candidate_embeddings, corpus_embeddings)
        else:
            corpus_texts = [build_embedding_text(c.title, c.abstract) for c in corpus]
            sim = self.get_similarity_score(candidate_texts, corpus_texts)
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10  # [n_candidate]
        for s, c in zip(scores, candidates):
            c.score = s
        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return candidates

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        """Embed both sides from scratch and return cosine similarity."""
        s1_embeddings = self.embed_texts(s1)
        s2_embeddings = self.embed_texts(s2)
        return self._cosine(s1_embeddings, s2_embeddings)


registered_rerankers = {}

def register_reranker(name: str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name: str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
