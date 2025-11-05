import numpy as np
from openai import OpenAI
from typing import List, Callable, Dict
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from src.common.faiss_manager import FAISSIndexManager
from src.common.llm.openai_atomicfact_generator import OpenAIAtomicFactGenerator
from src.subclaim_processor.scorer.document_scorer import IDocumentScorer
from src.subclaim_processor.strategies.aggregation import (
    AggregationStrategy,
    MaxAggregation,
    MeanAggregation,
)
from src.subclaim_processor.strategies.scoring import (
    ScoringStrategy,
    ProductScoreStrategy,
)

AGGREGATION_STRATEGIES: Dict[str, Callable] = {
    "max": MaxAggregation,
    "mean": MeanAggregation,
}

SCORING_STRATEGIES: Dict[str, Callable] = {
    "product": ProductScoreStrategy,
}


class SubclaimScorer(IDocumentScorer):
    def __init__(
        self,
        index_truncation_config,
        embedding_model="text-embedding-3-large",
        index_path="index_store/index.faiss",
        indice2fm_path="index_store/indice2fm.json",
    ):
        self.embedding_model = embedding_model
        self.faiss_manager = FAISSIndexManager(
            index_truncation_config=index_truncation_config,
            index_path=index_path,
            indice2fm_path=indice2fm_path,
        )
        self.gen = OpenAIAtomicFactGenerator()
        self.openai_client = OpenAI()

    def score(
        self,
        claim: str,
        retrieved_docs: List[Document],
        aggregation_strategy: AggregationStrategy,
        scoring_strategy: ScoringStrategy,
    ) -> float:

        if aggregation_strategy not in AGGREGATION_STRATEGIES:
            raise ValueError(
                f"Unknown aggregation strategy: {aggregation_strategy}. "
                f"Supported strategies are: {list(AGGREGATION_STRATEGIES.keys())}"
            )
        else:
            agg_func = AGGREGATION_STRATEGIES[aggregation_strategy]()

        if scoring_strategy not in SCORING_STRATEGIES:
            raise ValueError(
                f"Unknown scoring strategy: {scoring_strategy}. "
                f"Supported strategies are: {list(SCORING_STRATEGIES.keys())}"
            )
        else:
            scoring_func = SCORING_STRATEGIES[scoring_strategy]()

        if len(retrieved_docs) == 0:
            return 0

        claim_embedding = self.faiss_manager.openaiManager.client.embeddings.create(
            input=[claim], model=self.embedding_model
        )
        claim_vector = (
            np.array(claim_embedding.data[0].embedding).astype("float32").reshape(1, -1)
        )

        doc_scores = []
        for doc in retrieved_docs:
            parsed_doc = self.faiss_manager.parse_result(doc)
            doc_embedding = self.faiss_manager.index.reconstruct(parsed_doc["indice"])

            score = scoring_func.compute_score(
                claim_vector=claim_vector,
                doc_embedding=doc_embedding,
                parsed_doc=parsed_doc,
            )
            doc_scores.append(score)

        return 0 if len(retrieved_docs) == 0 else agg_func.aggregate(doc_scores)

    def cosine_similarity(self, claim: str, query: str) -> float:
        # claim score will be the maximum product of cosine similarity between the claim and the retrieved documents

        claim_embedding = self.faiss_manager.openaiManager.client.embeddings.create(
            input=[claim], model=self.embedding_model
        )
        claim_vector = (
            np.array(claim_embedding.data[0].embedding).astype("float32").reshape(1, -1)
        )

        query_embedding = self.faiss_manager.openaiManager.client.embeddings.create(
            input=[query], model=self.embedding_model
        )
        query_vector = (
            np.array(query_embedding.data[0].embedding).astype("float32").reshape(1, -1)
        )

        score = cosine_similarity(claim_vector, query_vector)[0][0]

        return score

    def frequency_score(
        self,
        response_agent,
        question: str,
        subclaim: str,
        retrived_docs: List[Document],
        temperature: float,
        n_samples: int,
    ) -> float:
        # Generate n_samples alternate outputs with temperature 1.0.

        chat_responses = response_agent.answer(
            question, retrived_docs, temperature=temperature, n_samples=n_samples
        )
        alternative_responses = [
            choice.message.content for choice in chat_responses.choices
        ]

        # Count the number of times the alternate outputs support the sub-claims (using LM).
        scores = []
        for response in alternative_responses:
            counting_prompt = (
                "You will get a claim and piece of text. Score whether the text supports, contradicts, or is unrelated to the claim. Directly return a SCORE with no explanation or other formatting. For the SCORE, return 1 for supports, -1 for contradicts, and 0 for unrelated. The claim is:\n"
                + subclaim
                + "\n\nThe text is:\n"
                + response
            )
            messages = [{"role": "user", "content": counting_prompt}]
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                temperature=0,
                n=1,
            )
            score_response = completion.choices[0].message.content
            try:
                scores.append(int(score_response))
            except Exception as ex:
                print(ex)
                print(score_response)

        return sum(scores) / len(scores)

    # def say_less(self, prompt, thresholds, model="gpt-4"):
    #     """
    #     say_less takes in the model prompt, generate output, breaks it down into subclaims, and removes sub-claims up to the threshold value.
    #     """
    #     output = ""
    #     retrieved_docs = self.faiss_manager.search_faiss_index(prompt, 10, 0.3)
    #     output = self.faiss_manager.generate_response_from_context(
    #         prompt, retrieved_docs
    #     )
    #     atomicFacts = self.gen.get_facts_from_text(output)
    #     subclaims_with_score = []
    #     for fact in atomicFacts:
    #         purefact = fact.rpartition(":")[0] if ":" in fact else fact
    #         score = self.score(purefact, retrieved_docs)
    #         # store purefact and score pair into list
    #         subclaims_with_score.append((purefact, score))
    #     accepted_subclaims_per_threshold = []
    #     mergerd_output_per_threshold = []
    #     for threshold in thresholds:
    #         accepted_subclaims = [
    #             subclaim for subclaim in subclaims_with_score if subclaim[1] > threshold
    #         ]
    #         mergerd_output = self.merge_subclaims(accepted_subclaims, model, prompt)
    #         accepted_subclaims_per_threshold.append(accepted_subclaims)
    #         mergerd_output_per_threshold.append(mergerd_output)
    #     return (
    #         output,
    #         mergerd_output_per_threshold,
    #         subclaims_with_score,
    #         accepted_subclaims_per_threshold,
    #     )

    # def default_merge_prompt(subclaims, prompt):
    #     claim_string = "\n".join(
    #         [str(i) + ": " + subclaim[0] for i, subclaim in enumerate(subclaims)]
    #     )
    #     return f"You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and try to use all facts as long as its possible. If no facts are given, reply to the instruction incorporating the fact that you don't know enough to fully respond. \n\nThe facts:\n{claim_string}\n\nThe instruction:\n{prompt}"

    # def merge_subclaims(
    #     self, subclaims, model, prompt, create_merge_prompt=default_merge_prompt
    # ):
    #     """
    #     Takes in a list of sub-claims like [('Percy Liang is a computer scientist.', 5.0), ...] and produces a merged output.
    #     """
    #     prompt = create_merge_prompt(subclaims, prompt)
    #     output = (
    #         self.query_model(prompt, model, max_tokens=1000, temperature=0)
    #         if subclaims
    #         else "Abstain."
    #     )
    #     return output
