from typing import Optional, Union

from src.common.faiss_manager import FAISSIndexManager
from src.common.multi_hop_retriever import MultiHopRetriever
from src.common.openai_query_rewriter import OpenAIQueryRewriter
from src.common.query_rewriter import QueryRewriter
from src.common.retriever import Retriever
from src.common.single_hop_retriever import SingleHopRetriever


def create_retriever(
    strategy: str,
    faiss_manager: FAISSIndexManager,
    truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
    truncate_by: Optional[str] = "\n",
    multi_hop_config: Optional[dict] = None,
    query_rewriter: Optional[QueryRewriter] = None,
) -> Retriever:
    """
    Create a retrieval strategy from configuration.

    Args:
        strategy:
            Retrieval strategy name. Supports "single_hop" and "multi_hop".
        faiss_manager:
            FAISS index manager used by the base retrieval strategy.
        truncation_strategy:
            Document-processing strategy used by the FAISS retriever.
        truncate_by:
            Delimiter used by document processing when applicable.
        multi_hop_config:
            Configuration for multi-hop retrieval.
        query_rewriter:
            Optional preconstructed query rewriter. This is useful for
            testing or for injecting a non-OpenAI implementation.

    Returns:
        Configured Retriever implementation.

    Raises:
        ValueError:
            If configuration is invalid or an unsupported strategy or
            query-rewriter provider is requested.
    """
    if not strategy:
        raise ValueError("A retrieval strategy must be provided.")

    normalized_strategy = strategy.strip().lower()

    base_retriever = SingleHopRetriever(
        faiss_manager=faiss_manager,
        truncation_strategy=truncation_strategy,
        truncate_by=truncate_by,
    )

    if normalized_strategy == "single_hop":
        return base_retriever

    if normalized_strategy == "multi_hop":
        config = multi_hop_config or {}

        max_hops = config.get("max_hops", 3)
        accumulate_documents = config.get("accumulate_documents", True)
        stop_if_no_new_documents = config.get(
            "stop_if_no_new_documents",
            True,
        )

        if max_hops > 1 and query_rewriter is None:
            rewriter_config = config.get("query_rewriter", {})
            provider = rewriter_config.get("provider", "openai").strip().lower()

            if provider == "openai":
                query_rewriter = OpenAIQueryRewriter(
                    model=rewriter_config.get(
                        "model",
                        "gpt-4o-mini",
                    ),
                    temperature=rewriter_config.get(
                        "temperature",
                        0.0,
                    ),
                    max_documents=rewriter_config.get(
                        "max_documents",
                        5,
                    ),
                    max_chars_per_document=rewriter_config.get(
                        "max_chars_per_document",
                        1200,
                    ),
                )
            else:
                raise ValueError(
                    f"Unsupported query rewriter provider: '{provider}'. "
                    "Currently supported providers are: openai."
                )

        return MultiHopRetriever(
            base_retriever=base_retriever,
            query_rewriter=query_rewriter,
            max_hops=max_hops,
            accumulate_documents=accumulate_documents,
            stop_if_no_new_documents=stop_if_no_new_documents,
        )

    raise ValueError(
        f"Unknown retrieval strategy: '{strategy}'. "
        "Supported strategies are: single_hop, multi_hop."
    )
