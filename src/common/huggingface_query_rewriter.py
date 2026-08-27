from typing import Optional

from src.common.llm.huggingface_runtime import HuggingFaceRuntime
from src.common.query_rewriter import QueryRewriter


class HuggingFaceQueryRewriter(QueryRewriter):
    """
    Hugging Face-backed query rewriter for multi-hop retrieval.

    The prompt, evidence bounding, and STOP semantics mirror
    OpenAIQueryRewriter so retrieval-strategy behavior remains comparable
    across model providers.
    """

    def __init__(
        self,
        model: str,
        runtime: HuggingFaceRuntime = None,
        temperature: float = 0.0,
        max_documents: int = 5,
        max_chars_per_document: int = 1200,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        if temperature < 0:
            raise ValueError("temperature must be non-negative.")

        if max_documents < 1:
            raise ValueError("max_documents must be at least 1.")

        if max_chars_per_document < 1:
            raise ValueError("max_chars_per_document must be at least 1.")

        self.model = model.strip()
        self.runtime = runtime or HuggingFaceRuntime.for_model(self.model)
        self.temperature = temperature
        self.max_documents = max_documents
        self.max_chars_per_document = max_chars_per_document

    def _format_evidence(self, retrieved_docs: list[str]) -> str:
        """
        Format a bounded amount of retrieved evidence for query rewriting.
        """
        selected_docs = retrieved_docs[-self.max_documents :]

        if not selected_docs:
            return "No retrieved evidence is available."

        formatted_docs = []

        for index, document in enumerate(selected_docs, start=1):
            document_text = str(document).strip()
            document_text = document_text[: self.max_chars_per_document]
            formatted_docs.append(f"[Document {index}]\n{document_text}")

        return "\n\n".join(formatted_docs)

    @staticmethod
    def _parse_query(text: str) -> Optional[str]:
        """
        Convert model output into either a next-hop query or a stop signal.
        """
        query = text.strip()

        if not query:
            return None

        if query.upper() == "STOP":
            return None

        if query.startswith("```") and query.endswith("```"):
            query = query[3:-3].strip()

        if len(query) >= 2 and query[0] == query[-1] and query[0] in {'"', "'"}:
            query = query[1:-1].strip()

        return query or None

    def __call__(
        self,
        original_query: str,
        current_query: str,
        retrieved_docs: list[str],
        next_hop: int,
    ) -> Optional[str]:
        """
        Generate a focused query for the next retrieval hop.
        """
        if not original_query or not original_query.strip():
            raise ValueError("original_query must be non-empty.")

        if not current_query or not current_query.strip():
            raise ValueError("current_query must be non-empty.")

        if next_hop < 2:
            raise ValueError("next_hop must be at least 2.")

        evidence = self._format_evidence(retrieved_docs)

        system_prompt = (
            "You generate search queries for multi-hop retrieval. "
            "Use the original question, the current retrieval query, and "
            "the evidence found so far to identify what information is still "
            "missing. Generate one concise search query that would retrieve "
            "the missing evidence needed to answer the original question. "
            "Do not answer the original question. "
            "Do not explain your reasoning. "
            "Return only the new search query. "
            "If no useful additional retrieval query is needed, return STOP."
        )

        user_prompt = (
            f"Original question:\n{original_query.strip()}\n\n"
            f"Current retrieval query:\n{current_query.strip()}\n\n"
            f"Evidence retrieved so far:\n{evidence}\n\n"
            f"Generate the retrieval query for hop {next_hop}."
        )

        generations = self.runtime.generate(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=self.temperature,
            n_samples=1,
            max_new_tokens=64,
            return_token_probabilities=False,
        )

        if len(generations) != 1:
            raise ValueError(
                "Query rewriting must return exactly one sequence. "
                f"Received {len(generations)}."
            )

        return self._parse_query(generations[0].text)
