import numpy as np

class VolatilitySurface:
    """
    Surface de volatilité implicite : matrice (T, K)
    """
    def __init__(self, strikes, maturities):
        self.strikes = strikes
        self.maturities = maturities
        self.surface = np.zeros((len(maturities), len(strikes)))

    def set_vol(self, i, j, vol):
        self.surface[i, j] = vol

    def get_vol(self, i, j):
        return float(self.surface[i, j])
