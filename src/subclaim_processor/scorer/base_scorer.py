import numpy as np
from typing import List
from abc import ABC, abstractmethod

class IScorer(ABC):
    @abstractmethod
    def score(self, claim: str) -> float:
        pass