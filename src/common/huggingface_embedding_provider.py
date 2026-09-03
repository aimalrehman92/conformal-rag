from typing import Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.common.embedding_provider import EmbeddingProvider


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    Local Hugging Face / SentenceTransformers implementation of the
    provider-neutral embedding interface.

    The model is loaded lazily so configuration validation does not
    require downloading or initializing the embedding model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        super().__init__(model_name=model_name)
        self.device = device
        self._model = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

        return self._model

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings locally with SentenceTransformers.
        """
        texts = list(texts)

        if not texts:
            raise ValueError(
                "At least one text is required for embedding."
            )

        cleaned_texts = [
            text.replace("\n", " ")
            for text in texts
        ]

        embeddings = self._get_model().encode(
            cleaned_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding provider must return a 2-D matrix."
            )

        if embeddings.shape[0] != len(cleaned_texts):
            raise ValueError(
                "Embedding provider returned a different number "
                "of embeddings than input texts."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embedding provider returned NaN or infinite values."
            )

        return embeddings.tolist()
