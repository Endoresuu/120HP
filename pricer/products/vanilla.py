import numpy as np
from .base import Option

class EuropeanCall(Option):
    def __init__(self, K: float, T: float):
        super().__init__(strike=K, maturity=T)

    def payoff(self, ST):
        return np.maximum(ST - self.K, 0.0)


class EuropeanPut(Option):
    def __init__(self, K: float, T: float):
        super().__init__(strike=K, maturity=T)

    def payoff(self, ST):
        return np.maximum(self.K - ST, 0.0)
