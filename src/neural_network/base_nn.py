from abc import ABC, abstractmethod
from typing import Tuple

class BaseNeuralNetwork(ABC):
    
    @abstractmethod
    def fitness(self, indi) -> Tuple:
        pass

    @abstractmethod
    def evaluate(self, x, y) -> Tuple:
        pass

    
