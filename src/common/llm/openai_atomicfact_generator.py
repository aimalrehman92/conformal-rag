import os
import math
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.utils import string_utils


class OpenAIAtomicFactGenerator(object):
    def __init__(self, model: str = "gpt-4o-mini"):
        dotenv_path = os.path.join(os.getcwd(), ".env")
        load_dotenv(dotenv_path)
        self.instruction = """Please breakdown the following input into a set of small, independent claims, and return the results as a single array of pairs in the format [CLAIM1; CLAIM2; CLAIM3; ...]. Do not include new lines. Make sure delimeter is always ";". The input is: """
        self.client = OpenAI()
        self.model = model

    def get_atomic_facts_from_paragraph(self, paragraph: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant to breakdown long knowledge intensive text into independent fact.",
                },
                {"role": "user", "content": self.instruction + paragraph},
            ],
            logprobs=True,
            top_logprobs=1,
        )

        response = completion.choices[0].message.content
        num_tokens = len(completion.choices[0].logprobs.content)
        log_prob = [
            (
                completion.choices[0].logprobs.content[i].token,
                completion.choices[0].logprobs.content[i].top_logprobs[0].logprob,
            )
            for i in range(num_tokens)
        ]

        return response, log_prob

    def get_contents_from_title(self, db, title):
        data_list = db.get_text_from_title(title)
        contents = ""
        for data in data_list:
            contents += data["text"]
        return contents

    # def get_facts_from_title(self, db, title, model):
    #     contents = self.get_contents_from_title(db=db, title=title)
    #     facts = []
    #     # reason to do this is because the text in db sometimes are break in middle of a whole content
    #     # so re-joined all items in data, then re-break by <s></s> tag
    #     paragraphs = string_utils.extract_tag_content(contents)

    #     for paragraph in paragraphs:
    #         response = self.get_atomic_facts_from_paragraph(paragraph, model=model)
    #         result = string_utils.extract_array_result(response)
    #         facts.extend(string_utils.extract_string_array(result))
    #         # print("array list get extracted from is: " + result)
    #     return facts

    def extract_subclaim_log_probs(self, log_prob_tuples: list) -> list:
        """
        Group token probabilities by semicolon-delimited subclaim.

        A semicolon may appear as its own token or be attached to text.
        Preserve any non-delimiter text rather than discarding the entire
        token when a semicolon is present.
        """
        current_subclaim = []
        subclaims = []

        for token, log_prob in log_prob_tuples:
            probability = math.exp(log_prob)
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

        return subclaims

    # def extract_subclaim_log_probs(self, log_prob_tuples):
    #     current_subclaim = []
    #     subclaims = []
    #     in_subclaim = False

    #     for token, log_prob in log_prob_tuples:
    #         # Detect start of a new subclaim
    #         if '{"' in token and not in_subclaim:
    #             in_subclaim = True
    #             continue

    #         # Detect end of subclaim
    #         if '"}' in token or '"]' in token or '.","' in token:
    #             if current_subclaim:
    #                 subclaims.append(current_subclaim)
    #                 current_subclaim = []
    #             in_subclaim = False
    #             continue

    #         # Skip tokens related to subclaim markers
    #         if token in ['sub', 'claim', '":["']:
    #             continue

    #         # If we're inside a subclaim, collect token and probability
    #         if in_subclaim:
    #             current_subclaim.append((token, log_prob))

    #     return subclaims

    # def preprocess_llm_response(self, response_text: str) -> list:
    #     """
    #     Convert jsonl formatted llm response into a list of strings.

    #     Args:
    #         response_text (str): original llm output formated as jsonl.

    #     Returns:
    #         list: a list of subclaims
    #     """
    #     clean_text = response_text.replace('```jsonl\n', '').replace('```', '')

    #     subclaims = []
    #     for line in clean_text.strip().split('\n'):
    #         if line.strip():
    #             try:
    #                 json_obj = json.loads(line)
    #                 if isinstance(json_obj.get('subclaim'), list):
    #                     subclaims.extend(json_obj['subclaim'])
    #             except json.JSONDecodeError:
    #                 continue

    #     return subclaims

    def get_facts_from_text(self, text):
        response, log_probs = self.get_atomic_facts_from_paragraph(text)
        subclaim_log_probs = self.extract_subclaim_log_probs(log_probs)
        # subclaims = self.preprocess_llm_response(response)
        result = string_utils.extract_array_result(response)
        subclaims = string_utils.extract_string_array(result)

        reduced_subclaim_log_probs = subclaim_log_probs
        while len(subclaims) != len(reduced_subclaim_log_probs):
            if len(reduced_subclaim_log_probs[-1]) == 1:
                print(f"removing last subclaim {reduced_subclaim_log_probs[-1][0][0]}")
                del reduced_subclaim_log_probs[-1]
            else:
                raise ValueError(
                    f"""facts list and subclaim_mean_log_probs list must have the same length. 
                    Fact count: {len(subclaims)}; 
                    log_prob Count: {len(reduced_subclaim_log_probs)}
                    """
                )

        return zip(subclaims, reduced_subclaim_log_probs)
