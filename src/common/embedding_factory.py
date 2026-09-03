from src.common.embedding_provider import EmbeddingProvider


def create_embedding_provider(
    provider: str,
    model_name: str,
) -> EmbeddingProvider:
    """
    Create the configured embedding backend.

    Provider imports are intentionally lazy so an unused backend does
    not initialize or require credentials.
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(
            "Embedding provider must be a non-empty string."
        )

    normalized_provider = provider.strip().lower()

    if normalized_provider == "huggingface":
        from src.common.huggingface_embedding_provider import (
            HuggingFaceEmbeddingProvider,
        )

        return HuggingFaceEmbeddingProvider(
            model_name=model_name,
        )

    if normalized_provider == "openai":
        from src.common.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )

        return OpenAIEmbeddingProvider(
            model_name=model_name,
        )

    raise ValueError(
        "Unknown embedding provider: "
        f"{provider!r}. Supported providers are: "
        "['huggingface', 'openai']."
    )
