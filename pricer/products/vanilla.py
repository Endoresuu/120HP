from .base import Option

class EuropeanCall(Option):
    def payoff(self, ST: float) -> float:
        return max(ST - self.K, 0.0)

class EuropeanPut(Option):
    def payoff(self, ST: float) -> float:
        return max(self.K - ST, 0.0)
