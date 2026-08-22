from .base import AggregationStrategy


class MaxAggregation(AggregationStrategy):
    def aggregate(self, scores):
        return max([0, *scores])
