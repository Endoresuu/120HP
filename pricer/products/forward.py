import numpy as np

class Forward:
    """
    Forward contract on an underlying.
    """

    def __init__(self, K: float, T: float):
        self.K = K
        self.T = T

    def forward_price(self, spot: float, r: float, q: float = 0.0):
        """
        Forward price F0
        """
        return spot * np.exp((r - q) * self.T)

    def value(self, spot: float, r: float, q: float = 0.0):
        """
        Present value of the forward contract
        """
        F0 = self.forward_price(spot, r, q)
        return np.exp(-r * self.T) * (F0 - self.K)
