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
from src.common.llm.openai_manager import OpenAIManager


class FAISSIndexManager:
    def __init__(
        self,
        index_truncation_config,
        dimension=3072,
        index_path="index_store/index.faiss",
        indice2fm_path="index_store/indice2fm.json",
    ):

        dotenv_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(dotenv_path)
        self.openaiManager = OpenAIManager()
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.file_managers = []
        self.indice2fm = (
            {}
        )  # Mapping from file texts tracking from file_path to faiss index indices, guarantee indice in asc order
        self.index_path = index_path
        self.indice2fm_path = indice2fm_path

        # initialize index and indice2fm from saved files
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded FAISS index from {index_path}")

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
        last_index_id = self.index.ntotal - 1
        return last_index_id == max(max(values) for values in self.indice2fm.values())

    def save_index(self, index_path, indice2fm_path):
        if self.index:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            # also save file_path to indice mapping, self.indice2fm should be updated before calling this function
            with open(indice2fm_path, mode="w") as file:
                json.dump(self.indice2fm, file, indent=4)

    def delete_index(self):
        self.index.reset()
        self.indice2fm = {}
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.indice2fm_path):
            os.remove(self.indice2fm_path)
        print("FAISS index deleted.")

    def upsert_file_to_faiss(
        self,
        file_manager,
        model="text-embedding-3-large",
        truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
        truncate_by: Optional[str] = "\n",
    ):
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
            embeddings = self.openaiManager.create_openai_embeddings(
                file_manager.texts, model=model
            )

            # Normalize embeddings
            embeddings_np = self.normalize_embeddings(embeddings)
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
        #faiss normalize give error zsh: segmentation fault python faiss manager at some edge case in hotpotqa
        #faiss.normalize_L2(embeddings_np)
        embeddings_normalized = normalize(embeddings_np, norm='l2', axis=1)
        return embeddings_normalized

    def search_faiss_index(
        self,
        query,
        top_k=10,
        threshold=0.5,
        truncation_strategy: Optional[Union[str, bool]] = "fixed_length",
        truncate_by: Optional[str] = "\n",
    ):
        if self.index.ntotal == 0:
            return []

        # Create a normalized embedding for the query
        query_embedding = self.normalize_embeddings(
            [
                self.openaiManager.client.embeddings.create(
                    input=[query], model="text-embedding-3-large"
                )
                .data[0]
                .embedding
            ]
        )[0].reshape(1, -1)

        # Perform the search
        similarity, indices = self.index.search(query_embedding, top_k)
        filtered_results = [
            (idx, similar)
            for idx, similar in zip(indices[0], similarity[0])
            if similar >= threshold
        ]
        results = []

        # Reverse map indices to file paths and text
        for idx, dist in filtered_results:
            file_path_found = None
            relative_idx = None

            # Find the file_path and relative index using self.indice2fm
            for file_path, indice_list in self.indice2fm.items():
                if idx in indice_list:
                    file_path_found = file_path
                    relative_idx = indice_list.index(idx)
                    break

            if file_path_found is not None and relative_idx is not None:
                # Find the corresponding file_manager
                file_manager = next(
                    (
                        fm
                        for fm in self.file_managers
                        if fm.file_path == file_path_found
                    ),
                    None,
                )

                if file_manager:
                    # Process the file if necessary
                    file_manager.process_document(
                        truncation_strategy=truncation_strategy, truncate_by=truncate_by
                    )
                    try:
                        # Get the text from the file_manager
                        text = file_manager.texts[relative_idx][
                            1
                        ]  # Assuming (index, text) tuples in file_manager.texts
                        results.append(
                            f"{text} indice={idx} fileposition={relative_idx} score={dist:.4f}"
                            # TODO reformat this
                            # {
                            #     "text": text,
                            #     "indice": idx,
                            #     "fileposition": relative_idx,
                            #     "score": round(dist, 4),
                            # }
                        )
                    except:
                        print(
                            f"Error while retriving id={relative_idx} from file manager. Skipping over id={relative_idx}."
                        )

                else:
                    results.append(
                        f"File manager not found for '{file_path_found}' score={dist:.4f}"
                    )
            else:
                # TODO reformat this
                results.append(f"Index not mapped, score={dist:.4f}")

        return results

    def parse_result(self, result):
        """
        Parse the result from the search and return the page content, metadata, indice, and score.
        """
        # Parse the input
        parsed_item = None
        pattern = re.compile(
            r"page_content='(.*?)'\smetadata=(\{.*?\})\sindice=(\d+)\sfileposition=(\d+)\sscore=([\d.]+)",
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

    def generate_response_from_context(self, query, retrieved_docs, model="gpt-4o"):
        if not retrieved_docs:
            return "No relevant documents found in the FAISS index."

        # Process retrieved documents into a clean context
        formatted_docs = []
        for doc in retrieved_docs:
            try:
                # Split the document string into page_content and metadata
                doc_parts = doc.split("metadata=")
                page_content = doc_parts[0].replace("page_content=", "").strip()
                metadata = (
                    doc_parts[1].strip() if len(doc_parts) > 1 else "Unknown source"
                )

                # Format each document clearly
                formatted_doc = f"Content: {page_content}\nSource: {metadata}"
                formatted_docs.append(formatted_doc)
            except Exception as e:
                formatted_docs.append(f"Error processing document: {e}")

        # Combine the formatted documents into a single context
        context = "\n\n---\n\n".join(formatted_docs)

        # Construct the prompt for the OpenAI API
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided context.",
            },
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": f"The following context was retrieved from the database:\n\n{context}",
            },
        ]

        # Generate response using OpenAI Chat API
        response = self.openaiManager.client.chat.completions.create(
            model=model, messages=messages, max_tokens=4096, temperature=0.7
        )
        return response.choices[0].message.content


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
    response = manager.generate_response_from_context(query, retrieved_docs)
    print(response)


if __name__ == "__main__":
    print("Running faiss_manager.py")
    main()
