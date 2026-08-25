from abc import ABC, abstractmethod
from typing import List, Any


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    """

    def __init__(self, document, chunk_size: int, overlap_size: int = 0):
        self.document = document
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    @abstractmethod
    def create_chunks(self) -> list[dict[str, Any]]:
        """
        Abstract method to be implemented by subclasses for chunking text.
        """
        pass


class FixedLengthChunker(BaseChunker):
    """
    Chunker that splits text into overlapping fixed-size chunks of words.
    """

    def create_chunks(self) -> list[str]:

        if self.chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be greater than 0, got {self.chunk_size}"
            )

        if self.overlap_size < 0:
            raise ValueError(
                f"overlap_size must be non-negative, got {self.overlap_size}"
            )

        if self.overlap_size >= self.chunk_size:
            raise ValueError(
                "overlap_size must be smaller than chunk_size. "
                f"Got chunk_size={self.chunk_size}, "
                f"overlap_size={self.overlap_size}."
            )

        chunks: list[str] = []

        text = self.document
        words = text.split()
        start = 0
        chunk_num = 0

        while start < len(words):
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.overlap_size
            chunk_num += 1

        return chunks, len(words)
