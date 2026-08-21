from typing import Optional, Union

from src.common.faiss_manager import FAISSIndexManager
from src.common.retriever import Retriever


class SingleHopRetriever(Retriever):
    """
    Single-hop retrieval strategy backed by a FAISS index.

    This class preserves the repository's existing one-query,
    one-retrieval-step behavior while exposing it through the
    provider-neutral Retriever interface.
    """

    def __init__(
        self,
        faiss_manager: FAISSIndexManager,
        truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
        truncate_by: Optional[str] = "\n",
    ):
        self.faiss_manager = faiss_manager
        self.truncation_strategy = truncation_strategy
        self.truncate_by = truncate_by

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[str]:
        """
        Retrieve relevant documents for a single query in one hop.
        """
        return self.faiss_manager.search_faiss_index(
            query=query,
            top_k=top_k,
            threshold=threshold,
            truncation_strategy=self.truncation_strategy,
            truncate_by=self.truncate_by,
        )
