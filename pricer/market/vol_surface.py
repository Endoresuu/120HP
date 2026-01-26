import numpy as np

class VolatilitySurface:
    """
    Data structure representing the Implied Volatility Surface (IVS).

    This class stores a discrete grid of implied volatilities indexed by
    strike prices and tenors (maturities). It acts as a container for the
    results generated during the calibration process.

    Attributes:
        strikes (array-like): List of strike prices (x-axis).
        maturities (array-like): List of times to maturity in years (y-axis).
        surface (np.ndarray): A 2D NumPy array of shape (len(maturities), len(strikes))
                              storing the calibrated volatility values.
    """
    def __init__(self, strikes, maturities):
        """
        Initializes an empty volatility surface grid.

        Args:
            strikes (array-like): The strike price levels.
            maturities (array-like): The maturity levels (tenors).
        """
        self.strikes = strikes
        self.maturities = maturities
        self.surface = np.zeros((len(maturities), len(strikes)))

    def set_vol(self, i, j, vol):
        """
        Assigns a volatility value to a specific grid point.

        Args:
            i (int): The row index corresponding to the maturity.
            j (int): The column index corresponding to the strike.
            vol (float): The implied volatility value to store.
        """
        self.surface[i, j] = vol

    def get_vol(self, i, j):
        """
        Retrieves the volatility value from a specific grid point.

        Args:
            i (int): The row index.
            j (int): The column index.

        Returns:
            float: The implied volatility at the specified grid intersection.
        """
        return float(self.surface[i, j])
