from BikerEnv import State
from numpy.random import random
class ValueFunctionApproximator:
    def __init__(self, n=10) -> None:
        self.weights = random(size=n)

    def predict(self, s, a=None):
        pass
