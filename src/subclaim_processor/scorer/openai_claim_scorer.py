import os
import re
from dotenv import load_dotenv
from openai import OpenAI


class OpenAIClaimScorer(object):
    def __init__(self, response_model: str):
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.instruction = f"""givern question $query,
                please help provide a float score between 0 and 1 to the following claim
                based on confidence level you think claim is relevant and itself is true to question
                The claim is:"""
        self.client = OpenAI()
        self.response_model = response_model

    def openAI_response(self, query, claim):
        content = self.instruction.replace('$query', query) + claim
        completion = self.client.chat.completions.create(
            model=self.response_model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant to verify claims factuality and give intrinsic score"
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        return completion.choices[0].message.content

    def detect_float(self, answer):
        # Regex pattern for a float between 0 and 1, or 1.0 exactly with many 0s
        pattern = r'\b(?:0?\.\d+|1\.0+)\b'

        match = re.search(pattern, answer)
        if match:
            return float(match.group())
        else:
            print("No float found.")
            return 0.0
        