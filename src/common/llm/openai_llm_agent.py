from openai import OpenAI
import os
from dotenv import load_dotenv
from src.common.llm.llm_agent import LLMAgent


class OpenAILLMAgent(LLMAgent):
    def __init__(self, instruction: str, model="gpt-4o-mini"):
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.instruction = instruction
        self.model = model
        self.client = OpenAI()

    def answer(self, question) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": question
                }
            ],
            timeout=600,
        )
        return completion.choices[0].message.content
    
    def preProcess(self, query):
        return query
    
    def postProcess(self, response):
        return response