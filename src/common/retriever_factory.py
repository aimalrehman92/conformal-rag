from typing import Optional, Union

from src.common.faiss_manager import FAISSIndexManager
from src.common.multi_hop_retriever import MultiHopRetriever, QueryRewriter
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
            Callable responsible for generating the next-hop query.
            Required when strategy is "multi_hop" and max_hops > 1.

    Returns:
        Configured Retriever implementation.

    Raises:
        ValueError:
            If required configuration is missing or the strategy is unknown.
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
            raise ValueError(
                "A query_rewriter is required for multi-hop retrieval "
                "when max_hops is greater than 1."
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
