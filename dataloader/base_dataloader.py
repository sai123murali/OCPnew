# Define an abstract base class for data loaders

from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass