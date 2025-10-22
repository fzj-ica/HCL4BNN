from abc import ABC, abstractmethod

class INeuralNetwork(ABC):
    

    @abstractmethod
    def fitness(self, indi):
        pass

    @abstractmethod
    def evaluate(self, x, y):
        pass

    
