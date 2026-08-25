import os
import re
from dotenv import load_dotenv
from openai import OpenAI


class OpenAIClaimVerification(object):
    def __init__(self, model: str = "gpt-4o-mini"):
        dotenv_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(dotenv_path)
        self.labels = ["supported", "irrelevant", "unverifiable", "nonefactual"]
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
        self.client = OpenAI()
        self.model = model

    """
    This function will prompt openai api to give an annotation to subclaim. To perform a zero-shot annotation, leave document empty.
    """

    def openAI_response(self, query, answer, documents, claim):
        content = (
            self.instruction.replace("$query", query)
            .replace("$answer", answer)
            .replace("$documents", documents)
            + claim
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant to verify claims.",
                },
                {"role": "user", "content": content},
            ],
        )
        return completion.choices[0].message.content

    def detect_label(self, answer):
        if not isinstance(answer, str):
            raise ValueError("Verifier response must be a string.")

        label_pattern = "|".join(self.labels)

        # Preferred machine-readable format produced by the verifier prompt.
        explicit_match = re.fullmatch(
            rf"\s*(?:final\s+)?label\s*:\s*({label_pattern})\s*[.!]?\s*",
            answer,
            re.IGNORECASE,
        )

        if explicit_match:
            label = explicit_match.group(1).lower()
            return self.annotations[self.labels.index(label)]

        # Accept a bare label for compatibility with concise model responses.
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

    def annotate(self, query, answer, documents, claim):
        response = self.openAI_response(query, answer, documents, claim)
        return self.detect_label(response)
