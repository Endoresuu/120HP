import numpy as np

class Forward:
    """
    Representation of a Forward contract on a financial asset.

    A forward contract is a customized derivative contract between two parties
    to buy or sell an asset at a specified price (strike) on a future date (maturity).
    Unlike options, it represents a commitment, not a right.

    Attributes:
        K (float): The delivery price (strike) agreed upon in the contract.
        T (float): Time to maturity in years.
    """

    def __init__(self, K: float, T: float):
        """
        Initializes the Forward contract with a strike price and maturity.

        Args:
            K (float): The delivery price.
            T (float): Time to maturity in years.
        """
        self.K = K
        self.T = T

    def forward_price(self, spot: float, r: float, q: float = 0.0):
        """
        Calculates the theoretical forward price (the 'fair' delivery price).

        The forward price is the delivery price that would make the contract's
        value zero today, based on the cost-of-carry model.

        Args:
            spot (float): Current price of the underlying asset.
            r (float): Annualized risk-free interest rate.
            q (float): Annualized continuous dividend yield.

        Returns:
            float: The theoretical forward price: F = S0 * exp((r - q) * T).
        """
        return spot * np.exp((r - q) * self.T)

    def value(self, spot: float, r: float, q: float = 0.0):
        """
        Calculates the current mark-to-market value of the forward contract.

        This represents the present value of the difference between the
        current theoretical forward price and the contract's delivery price.

        Args:
            spot (float): Current price of the underlying asset.
            r (float): Annualized risk-free interest rate.
            q (float): Annualized continuous dividend yield.

        Returns:
            float: The present value of the contract: exp(-r * T) * (F - K).
        """
        F0 = self.forward_price(spot, r, q)
        return np.exp(-r * self.T) * (F0 - self.K)
