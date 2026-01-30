import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricer.market.data import MarketData
from pricer.models.heston import HestonModel
from pricer.products.vanilla import EuropeanCall, EuropeanPut

# =====================================================
# --- FONCTIONS DE CALCUL CACHÉES ---
# =====================================================

@st.cache_data
def run_heston_simulation_cached(S0, r, T, K, opt_type,
                                 v0, kappa, theta, sigma_v, rho,
                                 n_steps, n_paths):
    """
    Effectue le calcul du prix et génère les trajectoires.
    On regroupe les deux pour s'assurer de la cohérence des données.
    """
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

    # ---- Pricing & Paths ----
    price = heston.price_european(option, market)
    paths = heston.simulate_paths(market, T)

    # On ne retourne qu'une seule trajectoire pour le graphique afin d'alléger le cache
    return price, paths[0]

# =====================================================
# --- FONCTION DE RENDU ---
# =====================================================

def render():
    st.subheader("Heston Monte Carlo")
    st.caption("Stochastic volatility pricing via Monte Carlo simulation")

    st.divider()

    # =====================================================
    # INPUTS
    # =====================================================
    col_inputs, col_outputs = st.columns([1.4, 1.6], gap="large")

    with col_inputs:
        st.subheader("Contract & Market")

        c1, c2 = st.columns(2)
        with c1:
            S0 = st.number_input("Spot S₀", value=100.0, min_value=1e-8, key="h_S0")
            K = st.number_input("Strike K", value=100.0, min_value=1e-8, key="h_K")
            T = st.number_input("Maturity T (years)", value=1.0, min_value=1e-6, key="h_T")

        with c2:
            r = st.number_input("Risk-free rate r", value=0.04, key="h_r")
            opt_type = st.radio("Option type", ["Call", "Put"], key="h_opt_type", horizontal=True)

        st.markdown("### Heston parameters")

        p1, p2 = st.columns(2)
        with p1:
            v0 = st.number_input("Initial variance v₀", value=0.04, min_value=1e-8, key="h_v0")
            kappa = st.number_input("Mean reversion κ", value=1.5, min_value=1e-8, key="h_kappa")
            theta = st.number_input("Long-run variance θ", value=0.04, min_value=1e-8, key="h_theta")

        with p2:
            sigma_v = st.number_input("Vol of variance σᵥ", value=0.3, min_value=1e-8, key="h_sigma_v")
            rho = st.number_input(
                "Correlation ρ",
                value=-0.7,
                min_value=-0.999,
                max_value=0.999,
                key="h_rho"
            )

        st.markdown("### Monte Carlo")

        m1, m2 = st.columns(2)
        with m1:
            n_steps = st.number_input("Time steps", value=100, min_value=1, step=1, key="h_steps")
        with m2:
            n_paths = st.number_input("MC paths", value=5000, min_value=100, step=100, key="h_paths")

        show_path = st.checkbox("Show one simulated path", value=True, key="h_show_path")

        run = st.button("Run Heston Monte Carlo", use_container_width=True, key="h_run")

    # =====================================================
    # OUTPUTS
    # =====================================================
    with col_outputs:
        st.subheader("Results")

        if run:
            with st.status("Running Monte Carlo Simulation...", expanded=True) as status:
                try:
                    price, sample_path = run_heston_simulation_cached(
                        S0, r, T, K, opt_type,
                        v0, kappa, theta, sigma_v, rho,
                        n_steps, n_paths
                    )

                    st.session_state.heston_data = {
                        "price": price,
                        "sample_path": sample_path
                    }

                    status.update(
                        label="Simulation complete",
                        state="complete",
                        expanded=False
                    )

                except Exception as e:
                    st.error(f"Heston Monte Carlo error: {e}")

        if "heston_data" in st.session_state:
            data = st.session_state.heston_data

            st.metric("Heston MC Price", f"{data['price']:.6f}")

            if show_path:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(data["sample_path"])
                ax.set_title("Sample Heston path (Spot)")
                ax.set_xlabel("Time step")
                ax.set_ylabel("S")
                ax.grid(True)
                st.pyplot(fig)
