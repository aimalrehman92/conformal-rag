from openai import OpenAI
import os
from dotenv import load_dotenv
from src.common.llm.llm_agent import LLMAgent


class OpenAIRAGAgent(LLMAgent):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
    ):
        dotenv_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(dotenv_path)

        self.instruction = (
            "You are a helpful assistant that answers questions "
            "based on provided context."
        )
        self.model = model
        self.client = OpenAI()

    def answer(
        self,
        question: str,
        retrieved_docs: list,
        temperature: float = 0.7,
        n_samples: int = 1,
    ) -> str:
        if len(retrieved_docs) == 0:
            print(
                f"No relevant documents found for the query '{question}'. Generating without context..."
            )

        # Process retrieved documents into a clean context
        formatted_docs = []
        for doc in retrieved_docs:
            try:
                # Split the document string into page_content and metadata
                doc_parts = doc.rsplit("metadata=", 1)

                page_content = doc_parts[0].strip()
                if page_content.startswith("page_content="):
                    page_content = page_content.removeprefix("page_content=").strip()

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
                "content": self.instruction,
            },
            {
                "role": "user",
                "content": (
                    "Use the following retrieved context to answer the question. "
                    "If the context does not contain enough information, say so.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ]

        # Generate response using OpenAI Chat API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=temperature,
            n=n_samples,
        )
        return response

    def preProcess(self, query):
        return query

    def postProcess(self, response):
        return response
