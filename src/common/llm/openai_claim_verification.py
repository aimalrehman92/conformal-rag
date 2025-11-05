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
                please help verify by any means including using internet 
                if the following claim can be labeled in following categories according to query and answer:
                {self.labels}
                Supported: If the claim is true and is relevant to infer the answer from query,
                Irrelevant: If the claim is true but irrelevant to answer and query,
                Unverifiable: If the claim is unverifiable,
                NoneFactual: Only if this claim is none factual. 
                The claim is:"""
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

        # Create a regex pattern to match the labels
        pattern = re.compile(r"\b(" + "|".join(self.labels) + r")\b", re.IGNORECASE)

        # Search for the first label in the answer
        match = pattern.search(answer)

        if match:
            # Find the index of the matched label and return the corresponding annotation
            label_index = self.labels.index(match.group(0).lower())
            return self.annotations[label_index]
        else:
            # Return 'NF' if no label is found
            return "NF"

    def annotate(self, query, answer, documents, claim):
        response = self.openAI_response(query, answer, documents, claim)
        return self.detect_label(response)
