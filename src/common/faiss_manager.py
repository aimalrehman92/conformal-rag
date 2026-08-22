import os
import json
import re
import ast
import faiss
from typing import Union, Optional
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import normalize
from src.common.file_manager import FileManager
from src.common.embedding_provider import EmbeddingProvider
from src.common.openai_embedding_provider import OpenAIEmbeddingProvider


class FAISSIndexManager:
    def __init__(
        self,
        index_truncation_config,
        dimension=None,
        embedding_model=None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        index_path="index_store/index.faiss",
        indice2fm_path="index_store/indice2fm.json",
    ):
        dotenv_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(dotenv_path)

        if embedding_provider is None:
            resolved_model = embedding_model or "text-embedding-3-large"
            embedding_provider = OpenAIEmbeddingProvider(
                model_name=resolved_model,
            )
        elif (
            embedding_model is not None
            and embedding_model != embedding_provider.model_name
        ):
            raise ValueError(
                "Embedding configuration mismatch: "
                f"embedding_model='{embedding_model}' but provider uses "
                f"'{embedding_provider.model_name}'."
            )

        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_provider.model_name

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) if dimension is not None else None

        self.file_managers = []
        self.indice2fm = {}

        self.index_path = index_path
        self.indice2fm_path = indice2fm_path

        # Initialize index and indice2fm from saved files.
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.dimension = self.index.d
            print(
                f"Loaded FAISS index from {index_path} "
                f"with dimension {self.dimension}"
            )

        if os.path.exists(indice2fm_path):
            with open(indice2fm_path, "r") as file:
                self.indice2fm = json.load(file)

            for file_path, _ in self.indice2fm.items():
                self.file_managers.append(
                    FileManager(
                        file_path=file_path,
                        index_truncation_config=index_truncation_config,
                    )
                )

    def is_indice_align(self):
        if self.index is None or self.index.ntotal == 0:
            return not self.indice2fm

        if not self.indice2fm:
            return False

        last_index_id = self.index.ntotal - 1
        return last_index_id == max(
            max(values) for values in self.indice2fm.values() if values
        )

    def save_index(self, index_path, indice2fm_path):
        if self.index is not None:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            # also save file_path to indice mapping, self.indice2fm should be updated before calling this function
            with open(indice2fm_path, mode="w") as file:
                json.dump(self.indice2fm, file, indent=4)

    def delete_index(self):
        self.index = None
        self.dimension = None
        self.indice2fm = {}
        self.file_managers = []

        if os.path.exists(self.index_path):
            os.remove(self.index_path)

        if os.path.exists(self.indice2fm_path):
            os.remove(self.indice2fm_path)

        print("FAISS index deleted.")

    def upsert_file_to_faiss(
        self,
        file_manager,
        model=None,
        truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
        truncate_by: Optional[str] = "\n",
    ):
        model = model or self.embedding_model

        if model != self.embedding_model:
            raise ValueError(
                "Embedding model mismatch: "
                f"FAISSIndexManager is configured for '{self.embedding_model}', "
                f"but indexing requested '{model}'."
            )

        if not file_manager.file_path in [
            file_manager.file_path for file_manager in self.file_managers
        ]:
            self.file_managers.append(file_manager)
        else:
            print(f"File '{file_manager.file_path}' already exists in the FAISS index.")
            return

        # Process the file if necessary
        # TODO: check if file_manager.texts will in any case be empty, if not, remove the below block
        if not file_manager.texts:
            print("Processing documents...")
            file_manager.process_document(
                truncation_strategy=truncation_strategy, truncate_by=truncate_by
            )
            print("Documents processing done.")

        # Generate embeddings and append to index if not already present
        if not file_manager.file_path in self.indice2fm:
            print("Creating embedding for the document...")
            document_texts = [text for _, text in file_manager.texts]

            embeddings = self.embedding_provider.embed_documents(document_texts)

            # Normalize embeddings
            embeddings_np = self.normalize_embeddings(embeddings)

            if embeddings_np.ndim != 2 or embeddings_np.shape[0] == 0:
                raise ValueError("Embedding provider returned no valid embeddings.")

            embedding_dimension = embeddings_np.shape[1]

            if self.index is None:
                self.dimension = embedding_dimension
                self.index = faiss.IndexFlatIP(self.dimension)

            elif self.index.d != embedding_dimension:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"FAISS index expects {self.index.d} dimensions, "
                    f"but model '{model}' returned {embedding_dimension}."
                )

            start_index = self.index.ntotal

            # Add embeddings to FAISS index
            self.index.add(embeddings_np)

            end_index = self.index.ntotal
            added_indices = list(range(start_index, end_index))

            # Update the self.indice2fm dictionary
            self.indice2fm[file_manager.file_path] = added_indices
            self.save_index(
                index_path=self.index_path, indice2fm_path=self.indice2fm_path
            )
            print(
                f"Embeddings from file '{file_manager.file_path}' added to FAISS index between indice {start_index} to {end_index}."
            )
        else:
            print(f"File '{file_manager.file_path}' already exists in the FAISS index.")

    def normalize_embeddings(self, embeddings):
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            raise ValueError("Embeddings contain NaNs or Infs.")
        embeddings_np = np.array(embeddings).astype("float32")
        # faiss normalize give error zsh: segmentation fault python faiss manager at some edge case in hotpotqa
        # faiss.normalize_L2(embeddings_np)
        embeddings_normalized = normalize(embeddings_np, norm="l2", axis=1)
        return embeddings_normalized

    def search_faiss_index(
        self,
        query,
        top_k=10,
        threshold=0.5,
        truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
        truncate_by: Optional[str] = "\n",
    ):

        if self.index is None or self.index.ntotal == 0:
            return []

        # Create a normalized embedding for the query using the same
        # embedding model that was configured for the FAISS index.
        query_vector = self.embedding_provider.embed_query(query)

        query_embedding = self.normalize_embeddings([query_vector])[0].reshape(1, -1)

        # Ensure the query embedding is compatible with the existing index.
        if query_embedding.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension mismatch: "
                f"FAISS index expects {self.index.d} dimensions, "
                f"but model '{self.embedding_model}' returned "
                f"{query_embedding.shape[1]}."
            )

        # Perform the search.
        similarity, indices = self.index.search(query_embedding, top_k)

        filtered_results = [
            (idx, similar)
            for idx, similar in zip(indices[0], similarity[0])
            if idx >= 0 and similar >= threshold
        ]

        results = []

        # Reverse map indices to file paths and text.
        for idx, dist in filtered_results:
            file_path_found = None
            relative_idx = None

            # Find the file path and relative index using self.indice2fm.
            for file_path, indice_list in self.indice2fm.items():
                if idx in indice_list:
                    file_path_found = file_path
                    relative_idx = indice_list.index(idx)
                    break

            if file_path_found is not None and relative_idx is not None:
                # Find the corresponding FileManager.
                file_manager = next(
                    (
                        fm
                        for fm in self.file_managers
                        if fm.file_path == file_path_found
                    ),
                    None,
                )

                if file_manager:
                    # Process the file if necessary.
                    file_manager.process_document(
                        truncation_strategy=truncation_strategy,
                        truncate_by=truncate_by,
                    )

                    try:
                        # file_manager.texts contains (index, text) tuples.
                        text = file_manager.texts[relative_idx][1]

                        results.append(
                            f"{text} "
                            f"indice={idx} "
                            f"fileposition={relative_idx} "
                            f"score={float(dist)!r}"
                        )
                    except (IndexError, TypeError):
                        print(
                            f"Error while retrieving id={relative_idx} "
                            f"from file manager. Skipping id={relative_idx}."
                        )

                else:
                    results.append(
                        f"File manager not found for '{file_path_found}' "
                        f"score={float(dist)!r}"
                    )

            else:
                results.append(f"Index not mapped, score={float(dist)!r}")

        return results

    def parse_result(self, result):
        """
        Parse the result from the search and return the page content, metadata, indice, and score.
        """
        # Parse the input
        parsed_item = None
        pattern = re.compile(
            r"page_content='(.*?)'\smetadata=(\{.*?\})\sindice=(\d+)\sfileposition=(\d+)\s"
            r"score=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
            re.DOTALL,
        )

        matches = pattern.findall(result)
        # assume only 1 row with matched pattern will be feed in each time, only remain last item
        for match in matches:
            page_content, metadata, indice, fileposition, score = match
            # Convert metadata string to a dictionary
            metadata_dict = ast.literal_eval(metadata)
            parsed_item = {
                "page_content": page_content.strip(),
                "metadata": metadata_dict,
                "indice": int(indice),
                "fileposition": int(fileposition),
                "score": float(score),
            }
        return parsed_item


def main():
    # Example Usage
    file_path1 = os.path.join(os.getcwd(), "documents", "2024_Corrective_RAGv2.pdf")
    file_manager1 = FileManager(file_path1)
    manager = FAISSIndexManager(dimension=3072)
    manager.upsert_file_to_faiss(file_manager1)

    file_path2 = os.path.join(os.getcwd(), "documents", "2023_Iterative_RGen.pdf")
    file_manager2 = FileManager(file_path2)
    manager.upsert_file_to_faiss(file_manager2)

    query = "tell me about corrective rag system."
    retrieved_docs = manager.search_faiss_index(query, top_k=10, threshold=0.1)
    print(retrieved_docs)


if __name__ == "__main__":
    print("Running faiss_manager.py")
    main()
