from typing import List

class NNEvaluator:
    """Evaluates individuals using a neural network."""
    def __init__(self, nn):
        self.nn = nn

    def __call__(self, individual: List[int]):
        return (self.nn.evaluate(individual),)
