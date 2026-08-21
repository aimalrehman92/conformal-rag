from abc import ABC, abstractmethod


class Retriever(ABC):
    """
    Provider-neutral interface for retrieving documents for a query.

    Retrieval strategies such as single-hop and multi-hop retrieval
    should implement this interface. Downstream generation and
    conformal-scoring code should depend on Retriever rather than
    directly depending on FAISS.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[str]:
        """
        Retrieve documents relevant to a query.

        Args:
            query: Query used for retrieval.
            top_k: Maximum number of documents to return.
            threshold: Minimum retrieval score required for a document.

        Returns:
            Retrieved documents in the repository's current string format.

        Notes:
            The string return format is retained temporarily for backward
            compatibility with the existing generation and scoring pipeline.
            A structured retrieval-result type can be introduced later
            without changing the retrieval-strategy abstraction itself.
        """
        raise NotImplementedError
