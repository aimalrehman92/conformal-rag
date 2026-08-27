from src.common.llm.huggingface_runtime import HuggingFaceRuntime
from src.utils import string_utils


class HuggingFaceAtomicFactGenerator:
    """
    Hugging Face-backed atomic-fact decomposer.

    The public get_facts_from_text() contract matches
    OpenAIAtomicFactGenerator:

        iterable[
            tuple[
                subclaim_text,
                list[tuple[token_text, token_probability]],
            ]
        ]

    This preserves the token-probability information required by the
    existing min_log_prob confidence score.
    """

    def __init__(
        self,
        model: str,
        runtime: HuggingFaceRuntime = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        self.instruction = (
            "Please breakdown the following input into a set of small, "
            "independent claims, and return the results as a single array "
            "of pairs in the format [CLAIM1; CLAIM2; CLAIM3; ...]. "
            'Do not include new lines. Make sure delimeter is always ";". '
            "The input is: "
        )

        self.model = model.strip()
        self.runtime = runtime or HuggingFaceRuntime.for_model(self.model)

    def get_atomic_facts_from_paragraph(
        self,
        paragraph: str,
    ) -> tuple[str, list[tuple[str, float]]]:
        """
        Generate semicolon-delimited atomic claims and their token
        probabilities.
        """
        if not isinstance(paragraph, str) or not paragraph.strip():
            raise ValueError("paragraph must be a non-empty string.")

        generations = self.runtime.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant to breakdown long "
                        "knowledge intensive text into independent fact."
                    ),
                },
                {
                    "role": "user",
                    "content": self.instruction + paragraph,
                },
            ],
            temperature=1.0,
            n_samples=1,
            max_new_tokens=512,
            return_token_probabilities=True,
        )

        if len(generations) != 1:
            raise ValueError(
                "Atomic-fact generation must return exactly one sequence. "
                f"Received {len(generations)}."
            )

        generation = generations[0]

        return generation.text, generation.token_probabilities

    @staticmethod
    def _clean_subclaim_token_group(
        token_group: list,
        strip_opening_bracket: bool = False,
        strip_closing_bracket: bool = False,
    ) -> list:
        """
        Remove response-formatting characters from a subclaim token group.

        This mirrors the cleanup behavior used by the OpenAI decomposer so
        formatting characters do not contribute to claim confidence.
        """
        cleaned_group = list(token_group)

        if not cleaned_group:
            return cleaned_group

        if strip_opening_bracket:
            while cleaned_group:
                token, probability = cleaned_group[0]
                cleaned_token = token.lstrip()

                if not cleaned_token.startswith("["):
                    break

                cleaned_token = cleaned_token[1:].lstrip()

                if cleaned_token:
                    cleaned_group[0] = (cleaned_token, probability)
                    break

                cleaned_group.pop(0)

        if strip_closing_bracket:
            while cleaned_group:
                token, probability = cleaned_group[-1]
                cleaned_token = token.rstrip()

                if not cleaned_token.endswith("]"):
                    break

                cleaned_token = cleaned_token[:-1].rstrip()

                if cleaned_token:
                    cleaned_group[-1] = (cleaned_token, probability)
                    break

                cleaned_group.pop()

        if not cleaned_group:
            return cleaned_group

        quote_pairs = [
            ('"', '"'),
            ("“", "”"),
            ("‘", "’"),
        ]

        first_token = cleaned_group[0][0].lstrip()
        last_token = cleaned_group[-1][0].rstrip()

        for opening_quote, closing_quote in quote_pairs:
            if first_token.startswith(opening_quote) and last_token.endswith(
                closing_quote
            ):
                token, probability = cleaned_group[0]
                cleaned_token = token.lstrip()[1:].lstrip()

                if cleaned_token:
                    cleaned_group[0] = (cleaned_token, probability)
                else:
                    cleaned_group.pop(0)

                if not cleaned_group:
                    break

                token, probability = cleaned_group[-1]
                cleaned_token = token.rstrip()[:-1].rstrip()

                if cleaned_token:
                    cleaned_group[-1] = (cleaned_token, probability)
                else:
                    cleaned_group.pop()

                break

        return cleaned_group

    def extract_subclaim_token_probabilities(
        self,
        token_probability_tuples: list[tuple[str, float]],
    ) -> list[list[tuple[str, float]]]:
        """
        Group generated token probabilities by semicolon-delimited claim.

        A semicolon may be its own token or occur inside a tokenizer piece.
        Non-delimiter text from the same token is preserved.
        """
        current_subclaim = []
        subclaims = []

        for token, probability in token_probability_tuples:
            token_parts = token.split(";")

            for part_index, part in enumerate(token_parts):
                if part:
                    current_subclaim.append((part, probability))

                is_delimiter = part_index < len(token_parts) - 1

                if is_delimiter:
                    if current_subclaim:
                        subclaims.append(current_subclaim)
                        current_subclaim = []

        if current_subclaim:
            subclaims.append(current_subclaim)

        cleaned_subclaims = []

        for index, subclaim in enumerate(subclaims):
            cleaned_subclaim = self._clean_subclaim_token_group(
                subclaim,
                strip_opening_bracket=index == 0,
                strip_closing_bracket=index == len(subclaims) - 1,
            )

            if cleaned_subclaim:
                cleaned_subclaims.append(cleaned_subclaim)

        return cleaned_subclaims

    def get_facts_from_text(self, text):
        """
        Return parsed atomic claims aligned with generated-token probabilities.
        """
        response, token_probabilities = self.get_atomic_facts_from_paragraph(text)

        subclaim_token_probabilities = self.extract_subclaim_token_probabilities(
            token_probabilities
        )

        result = string_utils.extract_array_result(response)
        subclaims = string_utils.extract_string_array(result)

        if len(subclaims) != len(subclaim_token_probabilities):
            raise ValueError(
                "Parsed subclaims and token-probability groups must have "
                "the same length. "
                f"Subclaim count: {len(subclaims)}; "
                "probability group count: "
                f"{len(subclaim_token_probabilities)}."
            )

        return zip(subclaims, subclaim_token_probabilities)
