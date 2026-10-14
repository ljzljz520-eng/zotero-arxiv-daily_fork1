from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np


@register_reranker("local")
class LocalReranker(BaseReranker):
    embedding_backend = "local"

    def __init__(self, config):
        super().__init__(config)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        from sentence_transformers import SentenceTransformer

        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        self._encoder = SentenceTransformer(
            self.config.reranker.local.model, trust_remote_code=True
        )
        return self._encoder

    def embedding_model_name(self) -> str:
        return self.config.reranker.local.model

    def embedding_model_params(self) -> dict:
        # encode_kwargs (task, prompt_name, dimensions, ...) change the
        # produced vectors, so they must participate in the cache fingerprint.
        return dict(self.config.reranker.local.get("encode_kwargs") or {})

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        encoder = self._get_encoder()
        encode_kwargs = self.config.reranker.local.get("encode_kwargs") or {}
        features = encoder.encode(
            texts, **encode_kwargs, show_progress_bar=bool(self.config.executor.debug)
        )
        # Raw features; BaseReranker applies cosine normalization, matching the
        # api reranker path.
        return np.asarray(features, dtype=np.float32)
