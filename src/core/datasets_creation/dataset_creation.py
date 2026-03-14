from abc import ABC, abstractmethod

# Abstract class for dataset creation
class DatasetCreation(ABC):
    def __init__(self, logger=None):
       self.logger = logger

    @abstractmethod    
    def create_dataset(self):
        pass
        