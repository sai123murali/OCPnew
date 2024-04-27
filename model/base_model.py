# Define an abstract base class for models

from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass