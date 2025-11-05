from .base import AggregationStrategy
import numpy as np

class MeanAggregation(AggregationStrategy):
    def aggregate(self, scores):
        return np.mean(scores)
