from .base import BaseReranker, register_reranker
from openai import OpenAI
import numpy as np


@register_reranker("api")
class ApiReranker(BaseReranker):
    embedding_backend = "api"

    def __init__(self, config):
        super().__init__(config)
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        # Lazily constructed so tests can monkeypatch the OpenAI symbol and
        # the HTTP client is reused across sync + rerank within one process.
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.reranker.api.key,
                base_url=self.config.reranker.api.base_url,
            )
        return self._client

    def embedding_model_name(self) -> str:
        return self.config.reranker.api.model

    def embedding_batch_size(self) -> int:
        return self.config.reranker.api.get("batch_size") or 64

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        client = self._get_client()
        batch_size = self.config.reranker.api.get("batch_size") or 64
        model = self.config.reranker.api.model
        batches: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=model)
            # Order by the response index: authoritative even if providers
            # reorder the data array.
            ordered = sorted(response.data, key=lambda r: r.index)
            batches.append(np.asarray([r.embedding for r in ordered], dtype=np.float32))
        return np.concatenate(batches, axis=0)
