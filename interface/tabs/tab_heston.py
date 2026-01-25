import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricer.market.data import MarketData
from pricer.models.heston import HestonModel
from pricer.products.vanilla import EuropeanCall, EuropeanPut

def render():

    st.subheader("Heston Monte Carlo")

    col1, col2 = st.columns(2)

    with col1:
        S0 = st.number_input("Spot S₀", value=100.0, min_value=1e-8, key="h_S0")
        r = st.number_input("Risk-free rate r", value=0.04, key="h_r")
        T = st.number_input("Maturity T (years)", value=1.0, min_value=1e-6, key="h_T")
        K = st.number_input("Strike K", value=100.0, min_value=1e-8, key="h_K")
        opt_type = st.radio("Option type", ["Call", "Put"], key="h_opt_type")

    with col2:
        v0 = st.number_input("Initial variance v₀", value=0.04, min_value=1e-8, key="h_v0")
        kappa = st.number_input("Mean reversion κ", value=1.5, min_value=1e-8, key="h_kappa")
        theta = st.number_input("Long-run variance θ", value=0.04, min_value=1e-8, key="h_theta")
        sigma_v = st.number_input("Vol of variance σᵥ", value=0.3, min_value=1e-8, key="h_sigma_v")
        rho = st.number_input("Correlation ρ", value=-0.7, min_value=-0.999, max_value=0.999, key="h_rho")
        n_steps = st.number_input("Time steps", value=100, min_value=1, step=1, key="h_steps")
        n_paths = st.number_input("MC paths", value=5000, min_value=100, step=100, key="h_paths")

    show_path = st.checkbox("Show one simulated path", value=True, key="h_show_path")

    if st.button("Run Heston Monte Carlo", key="h_run"):

        try:
            # ---- Market ----
            market = MarketData(spot=float(S0), r=float(r))

            # ---- Option ----
            option = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

            # ---- Model ----
            heston = HestonModel(
                kappa=kappa,
                theta=theta,
                sigma_v=sigma_v,
                rho=rho,
                v0=v0,
                n_steps=int(n_steps),
                n_paths=int(n_paths)
            )

            # ---- Pricing ----
            price = heston.price_european(option, market)
            st.success(f"Heston MC price = {price:.6f}")

            # ---- Plot ONE path only (optional) ----
            if show_path:
                S_paths = heston.simulate_paths(market, T)

                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(S_paths[0])
                ax.set_title("Sample Heston path (Spot)")
                ax.set_xlabel("Time step")
                ax.set_ylabel("S")
                ax.grid(True)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Heston Monte Carlo error: {e}")
