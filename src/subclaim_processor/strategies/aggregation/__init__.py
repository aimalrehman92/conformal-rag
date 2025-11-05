from .base import AggregationStrategy

from .max import MaxAggregation
from .mean import MeanAggregation

__all__ = ["AggregationStrategy", "MeanAggregation", "MaxAggregation"]