from typing import Sequence

from openai import OpenAI

from src.common.embedding_provider import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI implementation of the provider-neutral embedding interface.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        client=None,
    ):
        super().__init__(model_name=model_name)

        # Keep client creation lazy so offline configuration and tests
        # do not require an OpenAI API key.
        self._client = client

    def _get_client(self):
        """
        Lazily create and return the OpenAI client.
        """
        if self._client is None:
            self._client = OpenAI()

        return self._client

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for document texts using OpenAI.
        """
        if not texts:
            raise ValueError("At least one text is required for embedding.")

        cleaned_texts = [text.replace("\n", " ") for text in texts]

        response = self._get_client().embeddings.create(
            input=cleaned_texts,
            model=self.model_name,
        )

        embeddings = [item.embedding for item in response.data]

        if len(embeddings) != len(cleaned_texts):
            raise ValueError(
                "Embedding provider returned a different number of "
                "embeddings than input texts."
            )

        return embeddings
