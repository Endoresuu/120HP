from abc import ABC, abstractmethod

class Option(ABC):
    """
    Classe abstraite que toutes les options doivent hériter.
    """
    def __init__(self, strike: float, maturity: float):
        self.K = strike
        self.T = maturity

    @abstractmethod
    def payoff(self, ST: float) -> float:
        pass
