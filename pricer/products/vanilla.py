import numpy as np
from .base import Option

class EuropeanCall(Option):
    """
    Implementation of a European Call option contract.

    A European Call option gives the holder the right (but not the obligation)
    to buy an asset at the strike price (K) on a specific expiration date (T).

    The payoff is positive only if the terminal asset price (ST) is
    greater than the strike price.
    """
    def __init__(self, K: float, T: float):
        """
        Initializes the Call option using the base Option structure.

        Args:
            K (float): Strike price of the call.
            T (float): Time to maturity in years.
        """
        super().__init__(strike=K, maturity=T)

    def payoff(self, ST):
        """
        Calculates the Call payoff: max(ST - K, 0).

        Args:
            ST (float | np.ndarray): Price of the underlying at maturity.

        Returns:
            float | np.ndarray: The non-negative payoff.
        """
        return np.maximum(ST - self.K, 0.0)


class EuropeanPut(Option):
    """
    Implementation of a European Put option contract.

    A European Put option gives the holder the right (but not the obligation)
    to sell an asset at the strike price (K) on a specific expiration date (T).

    The payoff is positive only if the terminal asset price (ST) is
    less than the strike price.
    """
    def __init__(self, K: float, T: float):
        """
        Initializes the Put option using the base Option structure.

        Args:
            K (float): Strike price of the put.
            T (float): Time to maturity in years.
        """
        super().__init__(strike=K, maturity=T)

    def payoff(self, ST):
        """
        Calculates the Put payoff: max(K - ST, 0).

        Args:
            ST (float | np.ndarray): Price of the underlying at maturity.

        Returns:
            float | np.ndarray: The non-negative payoff.
        """
        return np.maximum(self.K - ST, 0.0)
