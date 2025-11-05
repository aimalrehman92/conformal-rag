from abc import ABC, abstractmethod

class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, scores) -> float:
        pass
