from abc import abstractmethod
from typing import List
from langchain.schema import Document
from src.subclaim_processor.scorer.base_scorer import IScorer


class IDocumentScorer(IScorer):

    @abstractmethod
    def score(self, claim: str, retrieved_docs: List[Document], **kwargs) -> float:
        pass
