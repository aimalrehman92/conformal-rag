from abc import ABC, abstractmethod

class LLMAgent(ABC):
    @abstractmethod
    def answer(self, question) -> str:
        pass

    @abstractmethod
    def preProcess(self, query):
        pass

    @abstractmethod
    def postProcess(self, response):
        pass