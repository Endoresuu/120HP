import numpy as np

import numpy as np

class Future:
    """
    Future contract on an underlying.
    Under constant interest rates and no convexity adjustment,
    the future price equals the forward price.
    """

    def __init__(self, T: float):
        self.T = T

    def future_price(self, spot: float, r: float, q: float = 0.0):
        """
        Future price F₀
        """
        return spot * np.exp((r - q) * self.T)

    def value(self, spot: float, r: float, q: float = 0.0):
        """
        Present value of the future contract.
        For a future, initial value is zero by construction.
        """
        return 0.0
