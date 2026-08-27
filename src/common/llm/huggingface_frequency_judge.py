from src.common.llm.huggingface_runtime import HuggingFaceRuntime


class HuggingFaceFrequencyJudge:
    """
    Hugging Face-backed support/contradiction judge used by the
    existing frequency confidence score.

    One claim/response pair is mapped strictly to:
        1  -> supports
        0  -> unrelated
        -1 -> contradicts
    """

    def __init__(
        self,
        model: str,
        runtime: HuggingFaceRuntime = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        self.model = model.strip()
        self.runtime = runtime or HuggingFaceRuntime.for_model(self.model)

    def score(
        self,
        claim: str,
        text: str,
    ) -> int:
        """
        Score whether text supports, contradicts, or is unrelated
        to claim using the same strict -1/0/1 semantics as the
        existing OpenAI frequency judge.
        """
        if not isinstance(claim, str):
            raise ValueError("claim must be a string.")

        if not isinstance(text, str):
            raise ValueError("text must be a string.")

        counting_prompt = (
            "You will get a claim and piece of text. "
            "Score whether the text supports, contradicts, or is unrelated "
            "to the claim. Directly return a SCORE with no explanation or "
            "other formatting. For the SCORE, return 1 for supports, "
            "-1 for contradicts, and 0 for unrelated. The claim is:\n"
            + claim
            + "\n\nThe text is:\n"
            + text
        )

        generations = self.runtime.generate(
            messages=[
                {
                    "role": "user",
                    "content": counting_prompt,
                }
            ],
            temperature=0.0,
            n_samples=1,
            max_new_tokens=16,
            return_token_probabilities=False,
        )

        if len(generations) != 1:
            raise ValueError(
                "Frequency judge must return exactly one sequence. "
                f"Received {len(generations)}."
            )

        score_response = generations[0].text

        if not isinstance(score_response, str):
            raise ValueError(
                "Frequency scorer response must be a string. "
                f"Received: {score_response!r}"
            )

        normalized_score = score_response.strip()

        if normalized_score not in {"-1", "0", "1"}:
            raise ValueError(
                "Frequency scorer must return exactly one of '-1', '0', or '1'. "
                f"Received: {score_response!r}"
            )

        return int(normalized_score)
