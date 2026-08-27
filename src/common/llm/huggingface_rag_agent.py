from dataclasses import dataclass

from src.common.llm.huggingface_runtime import HuggingFaceRuntime
from src.common.llm.llm_agent import LLMAgent


@dataclass(frozen=True)
class HuggingFaceMessage:
    content: str


@dataclass(frozen=True)
class HuggingFaceChoice:
    message: HuggingFaceMessage


@dataclass(frozen=True)
class HuggingFaceChatResponse:
    """
    Minimal compatibility wrapper matching the response shape consumed
    by the existing subclaim-processing pipeline.
    """

    choices: list[HuggingFaceChoice]


class HuggingFaceRAGAgent(LLMAgent):
    """
    Hugging Face-backed RAG response generator.

    The returned object intentionally exposes
    response.choices[i].message.content so existing pipeline callers do not
    need provider-specific branching.
    """

    def __init__(
        self,
        model: str,
        runtime: HuggingFaceRuntime = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        self.instruction = (
            "You are a helpful assistant that answers questions "
            "based on provided context."
        )
        self.model = model.strip()
        self.runtime = runtime or HuggingFaceRuntime.for_model(self.model)

    def answer(
        self,
        question: str,
        retrieved_docs: list,
        temperature: float = 0.7,
        n_samples: int = 1,
    ) -> HuggingFaceChatResponse:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        if len(retrieved_docs) == 0:
            print(
                f"No relevant documents found for the query "
                f"'{question}'. Generating without context..."
            )

        formatted_docs = []

        for doc in retrieved_docs:
            try:
                doc_parts = doc.rsplit("metadata=", 1)

                page_content = doc_parts[0].strip()
                if page_content.startswith("page_content="):
                    page_content = page_content.removeprefix("page_content=").strip()

                metadata = (
                    doc_parts[1].strip() if len(doc_parts) > 1 else "Unknown source"
                )

                formatted_docs.append(f"Content: {page_content}\nSource: {metadata}")
            except Exception as exc:
                formatted_docs.append(f"Error processing document: {exc}")

        context = "\n\n---\n\n".join(formatted_docs)

        messages = [
            {
                "role": "system",
                "content": self.instruction,
            },
            {
                "role": "user",
                "content": (
                    "Use the following retrieved context to answer the "
                    "question. If the context does not contain enough "
                    "information, say so.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ]

        generations = self.runtime.generate(
            messages=messages,
            temperature=temperature,
            n_samples=n_samples,
            max_new_tokens=4096,
            return_token_probabilities=False,
        )

        return HuggingFaceChatResponse(
            choices=[
                HuggingFaceChoice(
                    message=HuggingFaceMessage(
                        content=generation.text,
                    )
                )
                for generation in generations
            ]
        )

    def preProcess(self, query):
        return query

    def postProcess(self, response):
        return response
