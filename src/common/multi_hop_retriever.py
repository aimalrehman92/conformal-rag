import logging
import re
from typing import Optional

from src.common.query_rewriter import QueryRewriter
from src.common.retriever import Retriever

logger = logging.getLogger(__name__)


class MultiHopRetriever(Retriever):
    """
    Multi-hop retrieval strategy built on top of another Retriever.

    Each hop:
      1. retrieves evidence for the current query,
      2. accumulates previously unseen documents,
      3. asks the query rewriter for the next query,
      4. stops when the hop limit is reached or progress stalls.

    The underlying retriever can be FAISS-backed or replaced by another
    retrieval implementation without changing this class.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        query_rewriter: Optional[QueryRewriter],
        max_hops: int = 3,
        accumulate_documents: bool = True,
        stop_if_no_new_documents: bool = True,
    ):
        if max_hops < 1:
            raise ValueError("max_hops must be at least 1.")

        if query_rewriter is None and max_hops > 1:
            raise ValueError(
                "A query_rewriter is required when max_hops is greater than 1."
            )

        self.base_retriever = base_retriever
        self.query_rewriter = query_rewriter
        self.max_hops = max_hops
        self.accumulate_documents = accumulate_documents
        self.stop_if_no_new_documents = stop_if_no_new_documents

    @staticmethod
    def _normalize_query(query: str) -> str:
        """
        Normalize a query for repeated-query detection.
        """
        return " ".join(query.strip().lower().split())

    @staticmethod
    def _document_identity(document: str) -> str:
        """
        Return a stable identity for a retrieved document.

        FAISS retrieval currently appends a similarity score to the
        document string. The score may change across hops even when the
        underlying document is identical, so it must not participate in
        deduplication.
        """
        return re.sub(
            r"\s+score=[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*$",
            "",
            document,
        ).strip()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[str]:
        """
        Retrieve evidence across multiple query-reformulation hops.

        For multi-hop retrieval, top_k is applied to each individual hop.
        Documents are deduplicated across hops while preserving their
        first-seen order.
        """

        if not query or not query.strip():
            raise ValueError("A non-empty query must be provided.")

        original_query = query
        current_query = query

        seen_queries = set()
        seen_documents = set()
        accumulated_documents = []

        for hop_index in range(self.max_hops):
            hop_number = hop_index + 1

            logger.info(
                "Multi-hop retrieval hop %d/%d: query=%r",
                hop_number,
                self.max_hops,
                current_query,
            )

            normalized_query = self._normalize_query(current_query)

            if normalized_query in seen_queries:
                logger.info(
                    "Multi-hop retrieval stopped at hop %d: repeated query.",
                    hop_number,
                )
                break

            seen_queries.add(normalized_query)

            retrieved_docs = self.base_retriever.retrieve(
                query=current_query,
                top_k=top_k,
                threshold=threshold,
            )

            logger.info(
                "Multi-hop retrieval hop %d: retrieved %d document(s).",
                hop_number,
                len(retrieved_docs),
            )

            new_documents = []

            for doc in retrieved_docs:
                document_identity = self._document_identity(doc)

                if document_identity in seen_documents:
                    continue

                seen_documents.add(document_identity)
                new_documents.append(doc)

            logger.info(
                "Multi-hop retrieval hop %d: %d new document(s).",
                hop_number,
                len(new_documents),
            )

            if self.accumulate_documents:
                accumulated_documents.extend(new_documents)
            else:
                accumulated_documents = list(retrieved_docs)

            is_last_hop = hop_index == self.max_hops - 1

            if is_last_hop:
                logger.info(
                    "Multi-hop retrieval stopped: reached max_hops=%d.",
                    self.max_hops,
                )
                break

            if self.stop_if_no_new_documents and not new_documents:
                logger.info(
                    "Multi-hop retrieval stopped after hop %d: no new documents.",
                    hop_number,
                )
                break

            next_query = self.query_rewriter(
                original_query=original_query,
                current_query=current_query,
                retrieved_docs=list(accumulated_documents),
                next_hop=hop_index + 2,
            )

            logger.info(
                "Multi-hop retrieval hop %d: next query=%r",
                hop_number,
                next_query,
            )

            if next_query is None or not next_query.strip():
                logger.info(
                    "Multi-hop retrieval stopped after hop %d: "
                    "query rewriter requested stop.",
                    hop_number,
                )
                break

            if self._normalize_query(next_query) in seen_queries:
                logger.info(
                    "Multi-hop retrieval stopped after hop %d: "
                    "next query repeats a previous query.",
                    hop_number,
                )
                break

            current_query = next_query

        return accumulated_documents
