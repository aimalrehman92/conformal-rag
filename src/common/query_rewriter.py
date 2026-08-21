from abc import ABC, abstractmethod
from typing import Optional


class QueryRewriter(ABC):
    """
    Provider-neutral interface for multi-hop query rewriting.

    Implementations may use OpenAI, Hugging Face models, local Llama
    models, deterministic rules, or any other query-generation strategy.
    """

    @abstractmethod
    def __call__(
        self,
        original_query: str,
        current_query: str,
        retrieved_docs: list[str],
        next_hop: int,
    ) -> Optional[str]:
        """
        Generate the query for the next retrieval hop.

        Args:
            original_query:
                The user's original question.
            current_query:
                The query used for the most recent retrieval hop.
            retrieved_docs:
                Evidence accumulated so far.
            next_hop:
                One-based hop number that will use the generated query.
                For example, after hop 1 this value is 2.

        Returns:
            A new query string for the next hop, or None to stop retrieval.
        """
        raise NotImplementedError
