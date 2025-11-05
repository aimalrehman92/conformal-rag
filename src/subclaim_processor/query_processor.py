from abc import ABC, abstractmethod

class IQueryProcessor(ABC):
    # take query file path output query with subclaims
    
    @abstractmethod
    def generate_responses(self, query_file:str):
        pass
    
    @abstractmethod
    def get_subclaims_from_responses(self):
        pass

    # add score into subclaim file
    @abstractmethod
    def score_subclaim(self):
        pass

    # add annotation into subclaim file
    @abstractmethod
    def annotate_subclaim(self):
        pass