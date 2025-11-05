import os
import json
from typing import Union, Optional
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.common.chunker import FixedLengthChunker


class FileManager:
    def __init__(self, file_path: str, index_truncation_config: dict):
        self.file_path = file_path
        self.chunk_size = index_truncation_config["chunk_size"]
        self.chunk_overlap = index_truncation_config["chunk_overlap"]
        self.texts = []
        directory = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self.texts_file = os.path.join(directory, f"{base_name}_texts.json")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )  # TODO

        # Load texts from file if it exists
        if os.path.exists(self.texts_file):
            with open(self.texts_file, "r", encoding="utf-8-sig") as f:
                self.texts = json.load(f)
            print(f"Loaded texts from file: {self.texts_file}")

    def load_pdf_document(self):
        pdf_reader = PdfReader(self.file_path)
        documents = []

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:  # Ensure the page has text
                document = Document(
                    metadata={"source": self.file_path, "page": page_num},
                    page_content=page_text,
                )
                documents.append(document)

        return documents

    def dump_documents(self, texts):
        if texts and not os.path.exists(self.texts_file):
            with open(self.texts_file, "w") as f:
                json.dump(texts, f)
            print(f"Associated texts saved to file: {self.texts_file}")
        else:
            raise FileExistsError(
                f"File {self.texts_file} already exists. Please remove it before saving."
            )

    def process_pdf(self):
        data = self.load_pdf_document()

        documents = self.text_splitter.split_documents(data)
        self.texts = [(i, str(doc)) for i, doc in enumerate(documents)]
        self.dump_documents(self.texts)

    def process_document(
        self,
        truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
        chunk_size: int = 2000,
        overlap_size: int = 25,
        truncate_by: Optional[str] = "\n",
    ):
        """
        Process document according to the specified strategy.
        Either truncation_strategy or truncate_by must be provided, but not both.
        """
        if truncation_strategy is None and truncate_by is None:
            raise ValueError(
                "Either truncation_strategy or truncate_by must be provided"
            )

        if self.texts:
            return

        chunks = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for title, texts in data.items():
            if not truncation_strategy and not truncate_by:
                chunks.append(self.create_document(title, texts, self.file_path))
                print(f"{title} - No text splitting. Chunk size: {len(texts)}")
            elif truncation_strategy == "fixed_length":
                chunk_list = []
                for text in texts:
                    fixed_length_chunks, texts_word_cnt = FixedLengthChunker(
                        text, chunk_size, overlap_size
                    ).create_chunks()
                    chunk_list.extend(fixed_length_chunks)
                print(
                    f"Document '{title}' is splitted into {len(chunk_list)} chunk(s) by length of {chunk_size} words. Initial text size: {texts_word_cnt}."
                )
                for text in chunk_list:
                    if text.strip():
                        chunks.append(self.create_document(title, text, self.file_path))
            elif truncation_strategy == "recursive":  # Fixed typo in strategy name
                raise NotImplementedError(
                    "Recursive truncation is currently not supported"
                )
            else:
                # print("splitting by specific char")
                if isinstance(texts, str):
                    if truncate_by in texts:
                        split_texts = texts.split(truncate_by)
                    else:
                        split_texts = [texts]
                elif isinstance(texts, list):
                    split_texts = texts

                for text in split_texts:
                    if text.strip():
                        chunks.append(self.create_document(title, text, self.file_path))

        self.texts = [(i, str(doc)) for i, doc in enumerate(chunks)]
        self.dump_documents(self.texts)

    def create_document(self, title, text, file_path):
        """Create a document with the given title and text."""
        return Document(
            page_content=f"{title}: {text}",
            metadata={"source": title, "file_path": file_path},
        )
