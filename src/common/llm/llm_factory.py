from src.common.llm.huggingface_atomicfact_generator import (
    HuggingFaceAtomicFactGenerator,
)
from src.common.llm.huggingface_claim_verification import (
    HuggingFaceClaimVerification,
)
from src.common.llm.huggingface_rag_agent import HuggingFaceRAGAgent
from src.common.llm.openai_atomicfact_generator import (
    OpenAIAtomicFactGenerator,
)
from src.common.llm.openai_claim_verification import (
    OpenAIClaimVerification,
)
from src.common.llm.openai_rag_agent import OpenAIRAGAgent

SUPPORTED_LLM_PROVIDERS = {"openai", "huggingface"}


def _normalize_provider(provider: str) -> str:
    """
    Normalize and validate an LLM provider name.
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider must be a non-empty string.")

    normalized_provider = provider.strip().lower()

    if normalized_provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Supported providers are: "
            f"{sorted(SUPPORTED_LLM_PROVIDERS)}."
        )

    return normalized_provider


def create_rag_agent(
    provider: str,
    model: str,
    runtime=None,
):
    """
    Create the configured RAG response generator.
    """
    provider = _normalize_provider(provider)

    if provider == "openai":
        if runtime is not None:
            raise ValueError(
                "runtime injection is only supported for Hugging Face RAG agents."
            )

        return OpenAIRAGAgent(model=model)

    return HuggingFaceRAGAgent(
        model=model,
        runtime=runtime,
    )


def create_atomic_fact_generator(
    provider: str,
    model: str,
    runtime=None,
):
    """
    Create the configured atomic-fact decomposer.
    """
    provider = _normalize_provider(provider)

    if provider == "openai":
        if runtime is not None:
            raise ValueError(
                "runtime injection is only supported for "
                "Hugging Face atomic-fact generators."
            )

        return OpenAIAtomicFactGenerator(model=model)

    return HuggingFaceAtomicFactGenerator(
        model=model,
        runtime=runtime,
    )


def create_claim_verifier(
    provider: str,
    model: str,
    runtime=None,
):
    """
    Create the configured claim verifier.
    """
    provider = _normalize_provider(provider)

    if provider == "openai":
        if runtime is not None:
            raise ValueError(
                "runtime injection is only supported for "
                "Hugging Face claim verifiers."
            )

        return OpenAIClaimVerification(model=model)

    return HuggingFaceClaimVerification(
        model=model,
        runtime=runtime,
    )
