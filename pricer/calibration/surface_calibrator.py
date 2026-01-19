import numpy as np
import streamlit as st
from pricer.products.market_option import MarketOption
from pricer.market.vol_surface import VolatilitySurface
from pricer.calibration.implied_vol import NewtonImpliedVolSolver

class Calibrator:

    def __init__(self, strikes, maturities, S0, r, price_matrix):
        self.strikes = strikes
        self.maturities = maturities
        self.S0 = S0
        self.r = r
        self.price_matrix = price_matrix
        self.solver = NewtonImpliedVolSolver()

    def build_surface(self):
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
