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
    st.subheader("Heston Monte Carlo")

    # ===========================
    # Paramètres du modèle
    # ===========================
    col1, col2 = st.columns(2)

    with col1:
        S0 = st.number_input("Spot S₀", value=100.0, key="h_S0")
        r = st.number_input("Risk-free rate r", value=0.04, key="h_r")
        T = st.number_input("Maturity T (years)", value=1.0, key="h_T")
        K = st.number_input("Strike K", value=100.0, key="h_K")
        opt_type = st.radio("Option type", ["Call", "Put"], key="h_opt_type")

    with col2:
        v0 = st.number_input("Initial variance v₀", value=0.04, key="h_v0")
        kappa = st.number_input("Mean reversion κ", value=1.5, key="h_kappa")
        theta = st.number_input("Long-run variance θ", value=0.04, key="h_theta")
        sigma_v = st.number_input("Vol of variance σᵥ", value=0.3, key="h_sigma_v")
        rho = st.number_input("Correlation ρ", value=-0.7, key="h_rho")
        n_steps = st.number_input("Time steps", value=100, key="h_steps")
        n_paths = st.number_input("MC paths", value=5000, key="h_paths")

    if st.button("Run Heston simulation", key="h_run"):

        try:
            # ---- Market Data ----
            market = MarketData(spot=S0, r=r)

            # ---- Option ----
            option = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

            # ---- Heston model ----
            heston = HestonModel(
                v0=v0,
                kappa=kappa,
                theta=theta,
                sigma_v=sigma_v,
                rho=rho,
                n_steps=int(n_steps),
                n_paths=int(n_paths)
            )

            # ---- Pricing ----
            price_heston = heston.price_european(option, market)
            st.success(f"Heston MC price = {price_heston:.4f}")

            # ---- (Optionnel) trajectoires ----
            S_paths = heston.simulate_paths(market, option.T)

            # Exemple : afficher une trajectoire
            fig, ax = plt.subplots()
            ax.plot(S_paths[0, :])
            ax.set_title("Sample Heston path")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Spot")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in Heston Monte Carlo: {e}")