from abc import ABC, abstractmethod
from src.rag.retrieval import DocDB


class IRawDataProcessor(ABC):
    @abstractmethod
    # take input file path of raw data and output structured query data
    def get_queries(self, input_file: str, output_file: str):
        pass

    # take structured input query file path of structured query data and output raw data
    @abstractmethod
    def get_documents(self, query_file: str, output_file: str):
        pass


class DatasetProcessor(ABC):
    """Base abstract class for dataset processors"""

    @abstractmethod
    def process_queries(self, input_file: str, **kwargs) -> list:
        """Process queries from input file and return a list of formatted queries"""
        pass

    @abstractmethod
    def process_documents(self, query_file: str, db: DocDB, **kwargs) -> dict:
        """Process documents for queries and return a dictionary of documents"""
        pass
