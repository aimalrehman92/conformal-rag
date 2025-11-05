from .base import AggregationStrategy

class MaxAggregation(AggregationStrategy):
    def aggregate(self, scores):
        return max(scores)
