from typing import Optional, Union

from src.common.faiss_manager import FAISSIndexManager
from src.common.retriever import Retriever
from src.common.single_hop_retriever import SingleHopRetriever


def create_retriever(
    strategy: str,
    faiss_manager: FAISSIndexManager,
    truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
    truncate_by: Optional[str] = "\n",
    multi_hop_config: Optional[dict] = None,
) -> Retriever:
    """
    Create a retrieval strategy from configuration.

    Args:
        strategy:
            Retrieval strategy name. Currently supports "single_hop".
            "multi_hop" is reserved for the upcoming implementation.
        faiss_manager:
            FAISS index manager used by retrieval strategies.
        truncation_strategy:
            Document-processing strategy used by the current FAISS pipeline.
        truncate_by:
            Delimiter used by document processing when applicable.
        multi_hop_config:
            Configuration reserved for multi-hop retrieval.

    Returns:
        Configured Retriever implementation.

    Raises:
        NotImplementedError:
            If multi-hop retrieval is requested before its implementation
            is available.
        ValueError:
            If an unknown retrieval strategy is requested.
    """
    if not strategy:
        raise ValueError("A retrieval strategy must be provided.")

    normalized_strategy = strategy.strip().lower()

    if normalized_strategy == "single_hop":
        return SingleHopRetriever(
            faiss_manager=faiss_manager,
            truncation_strategy=truncation_strategy,
            truncate_by=truncate_by,
        )

    if normalized_strategy == "multi_hop":
        raise NotImplementedError(
            "Multi-hop retrieval is configured but has not been " "implemented yet."
        )

    raise ValueError(
        f"Unknown retrieval strategy: '{strategy}'. "
        "Supported strategies are: single_hop, multi_hop."
    )
