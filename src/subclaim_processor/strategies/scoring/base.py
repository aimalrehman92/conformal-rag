from abc import ABC, abstractmethod

class ScoringStrategy(ABC):
    @abstractmethod
    def compute_score(self, claim_vector, doc_embedding, parsed_doc) -> float:
        pass
