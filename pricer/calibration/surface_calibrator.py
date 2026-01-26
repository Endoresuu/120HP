import numpy as np
import streamlit as st
from pricer.products.market_option import MarketOption
from pricer.market.vol_surface import VolatilitySurface
from pricer.calibration.implied_vol import NewtonImpliedVolSolver

class Calibrator:
    """
    High-level engine to calibrate a complete Implied Volatility Surface.

    This class iterates over a grid of strikes and maturities, using market prices
    to populate a volatility matrix. It integrates with Streamlit to provide
    visual feedback during the calibration process.

    Attributes:
        strikes (array-like): List or array of strike prices.
        maturities (array-like): List or array of times to maturity (in years).
        S0 (float): Current price of the underlying asset.
        r (float): Annualized risk-free interest rate.
        price_matrix (np.ndarray): 2D array where element [i, j] is the market price
                                   for maturity i and strike j.
        solver (NewtonImpliedVolSolver): Numerical engine for IV inversion.
    """

    def __init__(self, strikes, maturities, S0, r, price_matrix):
        """
        Initializes the calibrator with the market price grid and asset parameters.
        """
        self.strikes = strikes
        self.maturities = maturities
        self.S0 = S0
        self.r = r
        self.price_matrix = price_matrix
        self.solver = NewtonImpliedVolSolver()

    def build_surface(self):
        """
        Iterates through the price matrix to solve for implied volatility at every point.

        The method populates a `VolatilitySurface` object by solving the Black-Scholes
        inversion for each strike/maturity pair. It includes a Streamlit progress bar
        to track the calibration status in real-time.

        Returns:
            VolatilitySurface: An object containing the calibrated IV grid and
                              metadata for the entire surface.

        Note:
            Uses a fixed initial guess (sigma0=0.25) for each point, which is
            generally stable for standard equity options.
        """
        vol_surface = VolatilitySurface(self.strikes, self.maturities)
        progress_bar1 = st.progress(0)

        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.strikes):
                price_mkt = self.price_matrix[i, j]
                opt = MarketOption(self.S0, K, T, self.r, price_mkt)

                sigma = self.solver.solve(opt, sigma0=0.25)
                vol_surface.set_vol(i, j, sigma)

            progress_bar1.progress((i + 1) / len(self.maturities))

        return vol_surface
