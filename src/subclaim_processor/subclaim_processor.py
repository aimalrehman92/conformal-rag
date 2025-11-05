import os
import json
import random
import logging
import numpy as np
from typing import Union, Optional
from tqdm import tqdm
from jsonschema import validate
from src.subclaim_processor.query_processor import IQueryProcessor
from src.common.llm.openai_rag_agent import OpenAIRAGAgent
from src.common.llm.openai_atomicfact_generator import OpenAIAtomicFactGenerator
from src.common.llm.openai_claim_verification import OpenAIClaimVerification
from src.subclaim_processor.scorer.base_scorer import IScorer
from src.subclaim_processor.scorer.document_scorer import IDocumentScorer
from src.calibration.utils import load_subclaim_data


class SubclaimProcessor(IQueryProcessor):
    def __init__(
        self,
        faiss_manager,
        response_model: str,
        fact_generation_model: str,
        claim_verification_model: str,
        scorer: IScorer,
        subclaims_file: str,
    ):
        self.faiss_manager = faiss_manager
        self.response_agent = OpenAIRAGAgent(faiss_manager, model=response_model)
        self.generator = OpenAIAtomicFactGenerator(model=fact_generation_model)
        self.verifier = OpenAIClaimVerification(model=claim_verification_model)
        print(f"claim_verification_model: {claim_verification_model}")
        self.scorer = scorer
        self.subclaims_file = subclaims_file
        with open(
            "data/out/subclaims_schema.json", "r", encoding="utf-8"
        ) as schemafile:
            self.subclaim_schema = json.load(schemafile)

    def generate_responses(
        self,
        query_file: str,
        top_k: int,
        threshold: float,
        response_temperature: float = 0.7,
        truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
        truncate_by: Optional[str] = "\n",
    ):
        """Generate responses for queries"""
        # Read queries
        with open(query_file, "r", encoding="utf-8") as jsonfile:
            queries = json.load(jsonfile)

        responses = []
        for query in tqdm(queries, desc="Generating responses"):
            question = query["input"]
            groups = query.get("groups", [])

            # Document retrieval
            retrieved_docs = self.faiss_manager.search_faiss_index(
                question,
                top_k=top_k,
                threshold=threshold,
                truncation_strategy=truncation_strategy,
                truncate_by=truncate_by,
            )

            # Generate response
            chat_response = self.response_agent.answer(
                question, retrieved_docs, temperature=response_temperature, n_samples=1
            )
            response = chat_response.choices[0].message.content

            responses.append(
                {
                    "query": question,
                    "gld_ans": query["output"]["answer"],
                    "retrieved_docs": retrieved_docs,
                    "response": response,
                    "groups": groups,
                    "subclaims": [],
                }
            )

        # Save responses
        with open(self.subclaims_file, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=4)
        print(f"Responses saved to {self.subclaims_file}")

    def get_subclaims_from_responses(self):
        """Process existing responses to extract subclaims and save updates incrementally"""
        # Read existing responses
        with open(self.subclaims_file, "r", encoding="utf-8") as jsonfile:
            queries = json.load(jsonfile)

        # Process each query and save updates in batches
        batch_size = 10
        for i in tqdm(range(0, len(queries), batch_size), desc="Extracting subclaims"):
            batch = queries[i : i + batch_size]
            modified = False

            # Process each query in the batch
            for query in batch:
                if query["response"] and not query.get(
                    "subclaims"
                ):  # Only process if no subclaims exist
                    try:
                        subclaims_with_log_probs = self.generator.get_facts_from_text(
                            query["response"]
                        )
                        query["subclaims"] = [
                            {
                                "subclaim": subclaim[0],
                                "scores": {
                                    "log_prob": [score for token, score in subclaim[1]]
                                },
                                "annotations": {},
                            }
                            for subclaim in subclaims_with_log_probs
                        ]
                        modified = True
                    except Exception as e:
                        print(
                            f"Error processing query: {query['query'][:50]}... Error: {str(e)}"
                        )
                        query["subclaims"] = (
                            []
                        )  # Add empty subclaims to mark as processed
                        modified = True

            # Save updates if any changes were made in this batch
            if modified:
                with open(self.subclaims_file, "w", encoding="utf-8") as f:
                    json.dump(queries, f, indent=4)
                    print(
                        f"Saved updates through batch ending at index {min(i + batch_size, len(queries))}"
                    )

        print(f"Completed subclaim extraction. Results saved in {self.subclaims_file}")

    def score_subclaim(self, aggregation_strategy, scoring_strategy):

        with open(self.subclaims_file, "r", encoding="utf-8") as jsonfile:
            subclaims_data = json.load(jsonfile)
            for entry in tqdm(subclaims_data, desc="Scoring subclaims"):
                validate(instance=entry, schema=self.subclaim_schema)
                for i, subclaim in enumerate(entry["subclaims"]):
                    if isinstance(self.scorer, IDocumentScorer):
                        if "noise" not in subclaim["scores"].keys():
                            subclaim["scores"]["noise"] = np.random.normal(0, 0.001)
                        if "relavance" not in subclaim["scores"].keys():
                            relavance_score = self.scorer.score(
                                claim=subclaim["subclaim"],
                                retrieved_docs=entry["retrieved_docs"],
                                aggregation_strategy=aggregation_strategy,
                                scoring_strategy=scoring_strategy,
                            )
                            subclaim["scores"]["relavance"] = float(relavance_score)
                        if (
                            "query_claim_cosine_similarity"
                            not in subclaim["scores"].keys()
                        ):
                            query_claim_cosine_similarity = (
                                self.scorer.cosine_similarity(
                                    subclaim["subclaim"], entry["query"]
                                )
                            )
                            subclaim["scores"]["query_claim_cosine_similarity"] = float(
                                query_claim_cosine_similarity
                            )
                        if (
                            "doc_claim_cosine_similarity"
                            not in subclaim["scores"].keys()
                        ):
                            doc_claim_cosine_similarities = []
                            for doc in entry["retrieved_docs"]:  # TODO
                                doc_claim_cosine_similarities.append(
                                    self.scorer.cosine_similarity(
                                        subclaim["subclaim"], doc
                                    )
                                )
                            subclaim["scores"]["doc_claim_cosine_similarity"] = (
                                float(max(doc_claim_cosine_similarities))
                                if doc_claim_cosine_similarities
                                else 0
                            )
                        if "frequency" not in subclaim["scores"].keys():
                            frequency_score = self.scorer.frequency_score(
                                response_agent=self.response_agent,
                                question=entry["query"],
                                subclaim=subclaim["subclaim"],
                                retrived_docs=entry["retrieved_docs"],
                                temperature=1,
                                n_samples=5,
                            )
                            subclaim["scores"]["frequency"] = float(frequency_score)
                        if "random" not in subclaim["scores"].keys():
                            subclaim["scores"]["random"] = random.random()
                        if "ordinal" not in subclaim["scores"].keys():
                            subclaim["scores"]["ordinal"] = (
                                (i / len(entry["subclaims"]))
                                if len(entry["subclaims"]) > 0
                                else 0
                            )
                        if (
                            "min_log_prob" not in subclaim["scores"].keys()
                            and "log_prob" in subclaim["scores"].keys()
                        ):
                            subclaim["scores"]["min_log_prob"] = min(
                                subclaim["scores"]["log_prob"]
                            )

        with open(self.subclaims_file, "w", encoding="utf-8") as jsonfile:
            json.dump(subclaims_data, jsonfile, indent=4)
        print(f"Subclaims with scores saved to {self.subclaims_file}.")

    def annotate_subclaim(self):
        with open(self.subclaims_file, "r", encoding="utf-8") as jsonfile:
            subclaims_data = json.load(jsonfile)

        batch_size = 10
        modified = False

        for i in tqdm(
            range(0, len(subclaims_data), batch_size),
            desc="Annotating subclaims in batches",
        ):
            batch = subclaims_data[i : i + batch_size]

            for entry in batch:
                try:
                    validate(instance=entry, schema=self.subclaim_schema)

                    # Skip if already annotated
                    if all(
                        subclaim.get("annotations", {}).get("gpt")
                        for subclaim in entry["subclaims"]
                    ):
                        continue

                    doc_contents = []
                    for doc in entry["retrieved_docs"]:
                        try:
                            # Split the document string into page_content and metadata
                            doc_parts = doc.split("metadata=")
                            page_content = (
                                doc_parts[0].replace("page_content=", "").strip()
                            )
                            doc_contents.append(page_content)
                        except Exception as e:
                            doc_contents.append(f"Error processing document: {e}")

                    # Combine the formatted documents into a single context
                    context = "\n".join(doc_contents)

                    for subclaim in entry["subclaims"]:
                        if not subclaim.get("annotations", {}).get(
                            "gpt"
                        ):  # Only annotate if not already done
                            gold_answer = (
                                " ".join(entry["gld_ans"])
                                if isinstance(entry["gld_ans"], list)
                                else entry["gld_ans"]
                            )
                            annotation = self.verifier.annotate(
                                entry["query"],
                                gold_answer,
                                context,
                                subclaim["subclaim"],
                            )
                            if "annotations" not in subclaim:
                                subclaim["annotations"] = {}
                            subclaim["annotations"]["gpt"] = annotation
                            modified = True

                except Exception as e:
                    logging.error(f"Error processing entry: {str(e)}")
                    continue

            # Save after each batch if there were modifications
            if modified:
                try:
                    with open(self.subclaims_file, "w", encoding="utf-8") as jsonfile:
                        json.dump(subclaims_data, jsonfile, indent=4)
                    logging.info(
                        f"Saved batch through index {min(i + batch_size, len(subclaims_data))}"
                    )
                    modified = False  # Reset modified flag
                except Exception as e:
                    logging.error(f"Error saving batch: {str(e)}")

        logging.info(f"Completed annotation. Results saved in {self.subclaims_file}")


def process_subclaims(
    query_path,
    subclaims_path,
    faiss_manager,
    scorer,
    config,
):

    truncation_strategy = config["index"]["truncation_config"]["strategy"]
    truncate_by = config["index"]["truncation_config"]["truncate_by"]

    top_k = config["rag"]["retrival_topk"]
    threshold = config["rag"]["retrival_threshold"]
    response_model = config["rag"]["response_model"]
    response_temperature = config["rag"]["response_temperature"]
    fact_generation_model = config["rag"]["fact_generation_model"]

    aggregation_strategy = config["conformal_prediction"]["aggregation_strategy"]
    scoring_strategy = config["conformal_prediction"]["scoring_strategy"]
    claim_verification_model = config["conformal_prediction"][
        "claim_verification_model"
    ]

    # Check if the file exists and load it if it does
    data = None
    if os.path.exists(subclaims_path):
        data = load_subclaim_data(subclaims_path)
        # Check if the data is valid
        score_method_to_check = [
            "noise",
            "relavance",
            "frequency",
            "query_claim_cosine_similarity",
            "doc_claim_cosine_similarity",
            "random",
            "ordinal",
            "min_log_prob",
        ]
        if data:
            needs_subclaim = any(len(pt["subclaims"]) == 0 for pt in data)

            if needs_subclaim:
                needs_scoring = True
                needs_annotation = True

            else:
                needs_scoring = any(
                    len(subclaim["scores"]) == 0
                    for pt in data
                    for subclaim in pt["subclaims"]
                ) or any(
                    score_method not in subclaim["scores"].keys()
                    for pt in data
                    for subclaim in pt["subclaims"]
                    for score_method in score_method_to_check
                )

                needs_annotation = any(
                    len(subclaim["annotations"]) == 0
                    for pt in data
                    for subclaim in pt["subclaims"]
                )

            if not (needs_subclaim or needs_scoring or needs_annotation):
                print(f"Subclaims data already exists in {subclaims_path}.")
                return data

    # Initialize processor only when needed
    processor = SubclaimProcessor(
        faiss_manager,
        response_model,
        fact_generation_model,
        claim_verification_model,
        scorer,
        subclaims_path,
    )

    # Generate subclaims if data doesn't exist
    if not data:
        processor.generate_responses(
            query_path,
            top_k=top_k,
            threshold=threshold,
            response_temperature=response_temperature,
            truncation_strategy=truncation_strategy,
            truncate_by=truncate_by,
        )
        processor.get_subclaims_from_responses()
        processor.score_subclaim(
            aggregation_strategy=aggregation_strategy, scoring_strategy=scoring_strategy
        )
        processor.annotate_subclaim()
    else:
        # Update only what's needed
        if needs_subclaim:
            processor.get_subclaims_from_responses()
        if needs_scoring:
            processor.score_subclaim(
                aggregation_strategy=aggregation_strategy,
                scoring_strategy=scoring_strategy,
            )
        if needs_annotation:
            processor.annotate_subclaim()

    return load_subclaim_data(subclaims_path)
