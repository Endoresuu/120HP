import streamlit as st
from pricer.market.data import MarketData
from pricer.models.black_scholes import BlackScholesModel
from pricer.models.heston import HestonModel
from pricer.pricing.engine import PricingEngine
from pricer.products.market_option import MarketOption
from pricer.products.vanilla import EuropeanCall, EuropeanPut
from pricer.calibration.market_calibrator import MarketSmileCalibrator
from pricer.calibration.surface_calibrator import Calibrator
from pricer.market.import_data import get_option_chain, get_close_price
from pricer.calibration.implied_vol import NewtonImpliedVolSolver
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour le 3D


import os
import sys

from interface.utils.market_helpers import (
    choose_next_expiry,
    choose_expiry_closest_to_T,
    get_market_call_price_from_chain
)

from pricer.models.bs_greeks import (
    delta_call, delta_put,
    gamma, vega,
    theta_call, theta_put,
    rho_call, rho_put
)

def render():

    st.subheader("Greeks")
    st.markdown("### 1) Greeks at a given point")

    col1, col2 = st.columns(2)

    with col1:
        S = st.number_input("Spot S₀", 100.0)
        K = st.number_input("Strike K", 100.0)
        T = st.number_input("Maturity T (years)", 1.0)

    with col2:
        r = st.number_input("Risk-free rate r", 0.04)
        sigma = st.number_input("Volatility σ", 0.20)
        opt_type = st.radio("Option type", ["Call", "Put"])
    if st.button("Compute Greeks"):

        if opt_type == "Call":
            delta = delta_call(S, K, r, sigma, T)
            theta = theta_call(S, K, r, sigma, T)
            rho   = rho_call(S, K, r, sigma, T)
        else:
            delta = delta_put(S, K, r, sigma, T)
            theta = theta_put(S, K, r, sigma, T)
            rho   = rho_put(S, K, r, sigma, T)

        gamma_val = gamma(S, K, r, sigma, T)
        vega_val  = vega(S, K, r, sigma, T)

        c1, c2, c3 = st.columns(3)
        c1.metric("Delta", f"{delta:.6f}")
        c2.metric("Gamma", f"{gamma_val:.6f}")
        c3.metric("Vega",  f"{vega_val:.6f}")

        c4, c5 = st.columns(2)
        c4.metric("Theta", f"{theta:.6f}")
        c5.metric("Rho",   f"{rho:.6f}")
    st.markdown("---")
    st.markdown("### 2) Greeks vs Strike")
    k_min = st.slider("Min K factor", 0.3, 1.0, 0.5)
    k_max = st.slider("Max K factor", 1.0, 2.0, 1.5)
    n_pts = st.slider("Grid points", 20, 300, 80)
    if st.button("Plot Greeks vs Strike"):

        K_grid = np.linspace(k_min * K, k_max * K, n_pts)

        delta_g = []
        gamma_g = []
        vega_g  = []

        for Ki in K_grid:
            if opt_type == "Call":
                delta_g.append(delta_call(S, Ki, r, sigma, T))
            else:
                delta_g.append(delta_put(S, Ki, r, sigma, T))

            gamma_g.append(gamma(S, Ki, r, sigma, T))
            vega_g.append(vega(S, Ki, r, sigma, T))
        fig, ax = plt.subplots()
        ax.plot(K_grid, delta_g, label="Delta")
        ax.plot(K_grid, gamma_g, label="Gamma")
        ax.plot(K_grid, vega_g,  label="Vega")
        ax.legend()
        ax.set_xlabel("Strike")
        ax.set_title("Greeks vs Strike")

        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 3) Gamma heatmap Γ(S, K)")
    s_min = st.slider("Min S factor", 0.5, 1.0, 0.7)
    s_max = st.slider("Max S factor", 1.0, 1.5, 1.3)
    s_min = st.slider("Min S factor", 0.5, 1.0, 0.7)
    s_max = st.slider("Max S factor", 1.0, 1.5, 1.3)
    if st.button("Plot Gamma Heatmap"):

        S_vals = np.linspace(s_min * S, s_max * S, 40)
        K_vals = np.linspace(0.7 * K, 1.3 * K, 40)

        G = np.zeros((len(S_vals), len(K_vals)))

        for i, Si in enumerate(S_vals):
            for j, Kj in enumerate(K_vals):
                G[i, j] = gamma(Si, Kj, r, sigma, T)
