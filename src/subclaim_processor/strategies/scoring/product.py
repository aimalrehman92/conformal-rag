from .base import ScoringStrategy
from sklearn.metrics.pairwise import cosine_similarity

class ProductScoreStrategy(ScoringStrategy):
    def compute_score(self, claim_vector, doc_embedding, parsed_doc):
        return parsed_doc["score"] * cosine_similarity(claim_vector, doc_embedding.reshape(1, -1))[0][0]
