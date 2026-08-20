from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    """
    Provider-neutral interface for generating text embeddings.

    Retrieval and scoring code should depend on this interface rather
    than directly depending on OpenAI, Hugging Face, or another backend.
    """

    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("An embedding model name must be provided.")

        self.model_name = model_name

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Embed a collection of document texts.

        Returns one embedding vector for each input text.
        """
        raise NotImplementedError

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Embed a single query.

        Providers may override this method if they use different
        document/query encoding behavior.
        """
        embeddings = self.embed_documents([text])

        if len(embeddings) != 1:
            raise ValueError(
                "Embedding provider must return exactly one vector "
                "for a single query."
            )

        return embeddings[0]
