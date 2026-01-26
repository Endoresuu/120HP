import numpy as np

import numpy as np

class Future:
    """
    Representation of a Financial Future contract.

    A Future is a standardized exchange-traded contract to buy or sell an asset
    at a future date. Unlike Forwards, Futures are settled daily, meaning
    gains and losses are realized every day through a margin account.

    Attributes:
        T (float): Time to maturity in years.
    """

    def __init__(self, T: float):
        """
        Initializes the Future contract with a specific maturity.

        Note:
            Unlike a Forward, a Future does not store a fixed delivery price 'K'
            as its value is reset to zero daily via the clearinghouse.

        Args:
            T (float): Time to maturity in years.
        """
        self.T = T

    def future_price(self, spot: float, r: float, q: float = 0.0):
        """
        Calculates the theoretical fair price of the Future contract.

        Based on the spot-future parity (cost-of-carry model).

        Args:
            spot (float): Current price of the underlying asset.
            r (float): Annualized risk-free interest rate.
            q (float): Annualized continuous dividend yield.

        Returns:
            float: The theoretical future price: F = S0 * exp((r - q) * T).
        """
        return spot * np.exp((r - q) * self.T)

    def value(self, spot: float, r: float, q: float = 0.0):
        """
        Returns the current mark-to-market value of the Future contract.

        Due to daily resettlement (marking-to-market), the value of a
        Futures contract is theoretically reset to zero at the end of
        each trading day.

        Returns:
            float: Always 0.0.
        """
        return 0.0
