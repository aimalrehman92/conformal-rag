import re

from src.common.llm.huggingface_runtime import HuggingFaceRuntime


class HuggingFaceClaimVerification:
    """
    Hugging Face-backed claim verifier.

    The public annotate() contract and strict label semantics match
    OpenAIClaimVerification.
    """

    def __init__(
        self,
        model: str,
        runtime: HuggingFaceRuntime = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        self.labels = [
            "supported",
            "irrelevant",
            "unverifiable",
            "nonefactual",
        ]
        self.annotations = ["S", "I", "U", "N"]

        self.instruction = f"""Given query $query and true answer $answer,
                with following supporting documents: $documents,
                please verify the following claim using only the query, true answer,
                and supporting documents provided below, and label it according to:
                {self.labels}
                Supported: If the claim is true and is relevant to infer the answer from query,
                Irrelevant: If the claim is true but irrelevant to answer and query,
                Unverifiable: If the claim is unverifiable,
                NoneFactual: Only if this claim is none factual.
                The claim is:
                Return exactly one final label in this format:
                LABEL: <supported|irrelevant|unverifiable|nonefactual>
                Do not include any explanation."""

        self.model = model.strip()
        self.runtime = runtime or HuggingFaceRuntime.for_model(self.model)

    def model_response(
        self,
        query: str,
        answer: str,
        documents: str,
        claim: str,
    ) -> str:
        """
        Generate the verifier's machine-readable label response.
        """
        content = (
            self.instruction.replace("$query", query)
            .replace("$answer", answer)
            .replace("$documents", documents)
            + f"\n{claim}"
        )

        generations = self.runtime.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant to verify claims.",
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            temperature=1.0,
            n_samples=1,
            max_new_tokens=64,
            return_token_probabilities=False,
        )

        if len(generations) != 1:
            raise ValueError(
                "Claim verification must return exactly one sequence. "
                f"Received {len(generations)}."
            )

        return generations[0].text

    def detect_label(self, answer: str) -> str:
        """
        Convert one strict final textual label to S/I/U/N.
        """
        if not isinstance(answer, str):
            raise ValueError("Verifier response must be a string.")

        label_pattern = "|".join(self.labels)

        explicit_match = re.fullmatch(
            rf"\s*(?:final\s+)?label\s*:\s*({label_pattern})\s*[.!]?\s*",
            answer,
            re.IGNORECASE,
        )

        if explicit_match:
            label = explicit_match.group(1).lower()
            return self.annotations[self.labels.index(label)]

        bare_match = re.fullmatch(
            rf"\s*({label_pattern})\s*[.!]?\s*",
            answer,
            re.IGNORECASE,
        )

        if bare_match:
            label = bare_match.group(1).lower()
            return self.annotations[self.labels.index(label)]

        raise ValueError(
            "Verifier response did not contain exactly one valid final label. "
            f"Response: {answer!r}"
        )

    def annotate(
        self,
        query: str,
        answer: str,
        documents: str,
        claim: str,
    ) -> str:
        response = self.model_response(
            query,
            answer,
            documents,
            claim,
        )
        return self.detect_label(response)
