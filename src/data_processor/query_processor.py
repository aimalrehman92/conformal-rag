import os
import json
import random
from jsonschema import RefResolver, validate
from collections import defaultdict

from src.rag.retrieval import DocDB
from src.data_processor.raw_data_processor import IRawDataProcessor
from src.data_processor.fact_score_processor import FactScoreProcessor
from src.data_processor.hotpot_qa_processor import HotpotQAProcessor
from src.data_processor.pop_qa_processor import PopQAProcessor
from src.data_processor.medlf_qa_processor import MedLFQAProcessor


class QueryProcessor(IRawDataProcessor):
    """Main query processor that delegates to specific dataset processors"""

    def __init__(
        self,
        db_path: str = "data/raw/WikiDB/enwiki-20230401.db",
        query_size: int = None,
    ):
        self.db = DocDB(db_path=db_path, data_path=None)
        self.dataset = None
        self.query_size = query_size
        self.processors = {
            "fact_score": FactScoreProcessor(),
            "hotpot_qa": HotpotQAProcessor(),
            "pop_qa": PopQAProcessor(),
            "medlf_qa": MedLFQAProcessor(),
        }

    def get_queries(
        self,
        dataset: str,
        input_file: str,
        output_dir: str,
        output_file: str,
        seed: int = 42,
    ):
        """
        Reads raw data from a file and extracts queries, storing them in a JSON file.
        Returns a dictionary mapping query inputs to their answers.

        Args:
            dataset: The name of the dataset to process
            input_file: Path to the input file with raw data
            output_file: Path where processed queries will be saved
            query_size: Number of queries to sample (None or -1 for all)
            seed: Random seed for reproducible sampling

        Returns:
            dict: A dictionary mapping query inputs to their answers
        """
        self.dataset = dataset
        self.input_file = input_file

        # Case 1: Output file already exists - load instead of process
        query_path = os.path.join(output_dir, output_file)
        if os.path.exists(query_path):
            print(f"{query_path} already exists.")
            with open(query_path, "r", encoding="utf-8") as jsonfile:
                queries = json.load(jsonfile)

        # Case 2: Output file doesn't exist - process and save
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Get the appropriate processor
            processor = self.processors.get(dataset)
            if not processor:
                raise ValueError(f"Unsupported dataset: {dataset}")

            # Process the queries
            queries = processor.process_queries(input_file)

            # Save processed queries
            with open(query_path, "w", encoding="utf-8") as jsonfile:
                json.dump(queries, jsonfile, indent=4)

            print(f"Queries saved to {output_file}")

        # Sample queries if needed
        if self.query_size and self.query_size != -1 and len(queries) > self.query_size:
            random.seed(seed)
            if "groups" in queries[0]:
                # Build group-to-queries mapping
                group_to_queries = defaultdict(list)
                for item in queries:
                    for group in item.get("groups", []):
                        group_to_queries[group].append(item)
                
                # Sort base on group size ascending
                group_sizes = {g: len(qs) for g, qs in group_to_queries.items()}
                sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1])  # (group, count)

                remaining_size = self.query_size
                group_allocation = {}

                # sample query from each group as even as possible
                # if smallest group has less items than required (e.g. sample 1000 quries from 5 groups, but smallest only have 100 items), take all
                # rest (900) will be in remaining size for other (4) groups, do it until the next group size is greater than sample required
                # First pass: allocate full group if it's too small
                remaining_groups = []
                for group, size in sorted_groups:
                    fair_share = remaining_size // (len(sorted_groups) - len(group_allocation)) if (len(sorted_groups) - len(group_allocation)) > 0 else 0
                    if size <= fair_share:
                        group_allocation[group] = size
                        remaining_size -= size
                    else:
                        remaining_groups.append(group)

                # Second pass: fair allocation among remaining groups
                for group in remaining_groups:
                    fair_share = remaining_size // (len(remaining_groups) - len([g for g in group_allocation if g in remaining_groups]))
                    allocated = min(fair_share, group_sizes[group])
                    group_allocation[group] = allocated
                    remaining_size -= allocated

                # Now sample
                sampled = []
                for group, count in group_allocation.items():
                    sampled.extend(random.sample(group_to_queries[group], count))  
                self.queries = sampled              
             
            else:
                self.queries = random.sample(queries, self.query_size)

            # Write the sampled queries back to the output file
            query_path = os.path.join(
                output_dir, f"sampled_{self.query_size}_{output_file}"
            )
            with open(query_path, "w", encoding="utf-8") as jsonfile:
                json.dump(self.queries, jsonfile, indent=4)

        else:
            self.queries = queries

        # Create input to answer mapping
        return {
            query["input"]: query["output"]["answer"] for query in self.queries
        }, query_path

    def get_documents(self, query_dir: str, output_dir: str, output_file: str) -> str:
        """
        Reads structured query data from a JSON file and generates a corresponding document list.

        Args:
            query_dir: Directory containing query data.
            output_dir: Directory to save the output file.
            output_file: Name of the output file.

        Returns:
            Path to the output file.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Construct output path
        output_path = os.path.join(
            output_dir, f"sampled_{self.query_size}_{output_file}"
        )

        # Return if output file already exists
        if os.path.exists(output_path):
            print(f"{output_path} already exists.")
            return output_path

        # Validate processor exists for the dataset
        processor = self.processors.get(self.dataset)
        if not processor:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # Validate schema for specific datasets
        if self.dataset in ["fact_score", "hotpot_qa", "pop_qa"]:
            for query in self.queries:
                self._validate_schema(query)

        # Determine queries to use
        queries_to_use = None
        if self.query_size and self.query_size != -1:
            queries_to_use = self.queries

        # Process documents
        documents = processor.process_documents(
            query_dir, self.db, queries_to_use, raw_query_dir=self.input_file
        )

        # Save documents to output file
        with open(output_path, "w", encoding="utf-8") as jsonfile:
            json.dump(documents, jsonfile, indent=4, ensure_ascii=False)

        print(f"Document list saved to {output_path}.")
        return output_path

    def _validate_schema(self, query: dict):
        """Validate a query against schema"""
        base_schema = None
        wiki_schema = None
        with open(
            "data/processed/base_schema.json", "r", encoding="utf-8"
        ) as schemafile:
            base_schema = json.load(schemafile)

        with open(
            "data/processed/wiki_schema.json", "r", encoding="utf-8"
        ) as schemafile:
            wiki_schema = json.load(schemafile)

        resolver = RefResolver("data/processed/base_schema.json", base_schema)
        validate(instance=query, schema=wiki_schema, resolver=resolver)


if __name__ == "__main__":
    # wiki_query_processor = QueryProcessor(db_path="data/raw/WikiDB/enwiki-20230401.db")
    # wiki_query_processor.get_queries(dataset="fact_score", input_file="data/raw/FactScore/raw_fact_score.json", output_file="data/processed/FactScore/fact_score_queries.json")
    # wiki_query_processor.get_documents(query_dir="data/processed/FactScore/fact_score_queries.json", output_file="data/processed/FactScore/fact_score_documents.txt")

    # wiki_query_processor = QueryProcessor(db_path="data/raw/WikiDB/enwiki-20230401.db")
    # wiki_query_processor.get_queries(dataset="hotpot_qa", input_file="data/raw/HotpotQA/raw_hotpot_qa.json", output_file="data/processed/HotpotQA/hotpot_qa_queries.json")
    # wiki_query_processor.get_documents(query_dir="data/processed/HotpotQA/hotpot_qa_queries.json", output_file="data/processed/HotpotQA/hotpot_qa_documents.txt")

    # wiki_query_processor = QueryProcessor(db_path="data/raw/WikiDB/enwiki-20230401.db")
    # wiki_query_processor.get_queries(dataset="pop_qa", input_file="data/raw/PopQA/raw_pop_qa.json", output_file="data/processed/PopQA/pop_qa_queries.json")
    # wiki_query_processor.get_documents(query_dir="data/processed/PopQA/pop_qa_queries.json", output_file="data/processed/PopQA/pop_qa_documents.txt")

    medlf_query_processor = QueryProcessor(db_path="data/raw/WikiDB/enwiki-20230401.db")
    medlf_query_processor.get_queries(
        dataset="medlf_qa",
        input_file="data/raw/MedLFQA",
        output_file="data/processed/MedLFQA/medlf_qa_queries.json",
    )
    medlf_query_processor.get_documents(
        query_dir="data/processed/MedLFQA/medlf_qa_queries.json",
        output_file="data/processed/MedLFQA/medlf_qa_documents.txt",
    )
