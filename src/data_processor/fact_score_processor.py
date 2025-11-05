import json

from src.rag.retrieval import DocDB
from src.utils.string_utils import extract_tag_content
from src.data_processor.raw_data_processor import DatasetProcessor


class FactScoreProcessor(DatasetProcessor):
    def process_queries(self, input_file: str) -> list:
        queries = []
        with open(input_file, "r", encoding="utf-8") as file:
            for line in file:
                title = line.strip()
                query = {
                    "input": f"What is {title}'s occupation?",
                    "output": {"answer": "", "provenance": [{"title": title}]},
                }
                queries.append(query)
        return queries

    def process_documents(
        self, query_file: str, db: DocDB, queries: dict = None, **kwargs
    ) -> dict:
        documents = {}

        # if sampled queries are provided, use them instead of the queries in the query_file
        # however, for medlfqa, the query file is mandatory
        if queries is None:
            with open(query_file, "r", encoding="utf-8") as jsonfile:
                queries = json.load(jsonfile)

        with open(query_file, "r", encoding="utf-8") as jsonfile:
            queries = json.load(jsonfile)
            for query in queries:
                title = query["output"]["provenance"][0]["title"]
                document = self._get_documents_per_query(title, db)
                documents[title] = document
        return documents

    def _get_documents_per_query(self, title: str, db: DocDB) -> list:
        """Returns a list of documents for a given query."""
        contents = ""
        try:
            docs = db.get_text_from_title(title)
            for data in docs:
                contents += data["text"]
            return extract_tag_content(contents)
        except Exception as e:
            print(f"Error retrieving documents for title {title}: {e}")
            return []
