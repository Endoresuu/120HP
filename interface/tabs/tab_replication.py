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

def render():
    st.header("Replication & Hedging playground (Black–Scholes)")

    # ----------------------------
    # Inputs
    # ----------------------------
    col1, col2 = st.columns(2)

    with col1:
        S0 = st.number_input("Spot S₀", value=100.0, key="rep_S0")
        K  = st.number_input("Strike K", value=100.0, key="rep_K")
        T  = st.number_input("Maturity T (years)", value=1.0, min_value=1e-4, key="rep_T")
        opt_type = st.radio("Option", ["Call", "Put"], key="rep_type")

    with col2:
        r = st.number_input("Risk-free rate r", value=0.04, key="rep_r")
        sigma = st.number_input("Volatility σ", value=0.20, min_value=1e-4, key="rep_sigma")
        q = st.number_input("Dividend yield q (optional)", value=0.0, key="rep_q")  # si tu ne gères pas q, laisse 0

    st.markdown("---")

    # ----------------------------
    # Choose "hedge spot" (where delta is computed)
    # ----------------------------
    st.subheader("Static replication (one delta computed at a chosen spot)")

    hedge_spot_mode = st.radio(
        "Delta computed at:",
        ["Current spot S₀", "Custom spot"],
        key="rep_hedge_spot_mode"
    )

    if hedge_spot_mode == "Current spot S₀":
        S_hedge = float(S0)
        st.info(f"Delta will be computed at S = {S_hedge:.4f}")
    else:
        S_hedge = st.number_input("Custom hedge spot", value=float(S0), min_value=1e-8, key="rep_Shedge")

    # ----------------------------
    # Compute option price & delta (use your engine for consistency)
    # ----------------------------
    # Build market & option
    market = MarketData(spot=S0, r=r, q=q) if "q" in MarketData.__init__.__code__.co_varnames else MarketData(spot=S0, r=r)

    option = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

    model = BlackScholesModel(market_data=market, sigma=sigma)
    engine = PricingEngine(model=model)

    # Option price today
    price0 = engine.price_european(option, kind=opt_type.lower())

    # Delta at hedge spot (use your greeks functions if you have them)
    # NOTE: I assume your greek signature is delta_call(S, K, r, sigma, T)
    if opt_type == "Call":
        delta0 = float(delta_call(S_hedge, K, r, sigma, T))
    else:
        delta0 = float(delta_put(S_hedge, K, r, sigma, T))

    # Build the static replicating portfolio:
    # Hold Δ shares and a cash position B so that:
    #   V0 = Δ*S_hedge + B = option_price0
    # => B = option_price0 - Δ*S_hedge
    B0 = price0 - delta0 * S_hedge

    # Terminal value of the portfolio at maturity (cash grows at r):
    #   V_T = Δ*S_T + B0*exp(rT)
    # (ignoring q in the cash account; for a student project, this is fine)
    # If you want to be “extra correct” with dividends, we can refine, but keep it simple.

    # ----------------------------
    # Display key numbers
    # ----------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Option price (t=0)", f"{price0:.6f}")
    c2.metric("Delta used (static)", f"{delta0:.6f}")
    c3.metric("Cash position B₀", f"{B0:.6f}")

    st.caption(
        "Static replication means you hedge once with Δ at the chosen spot. "
        "It will *not* match the payoff globally because the option is convex (Gamma)."
    )

    # ----------------------------
    # Payoff plot at maturity
    # ----------------------------
    st.subheader("Payoff comparison at maturity")

    s_min = st.number_input("Min S_T (plot)", value=0.5 * S0, min_value=1e-8, key="rep_smin")
    s_max = st.number_input("Max S_T (plot)", value=1.5 * S0, min_value=1e-8, key="rep_smax")
    n_pts = st.slider("Grid points", min_value=100, max_value=2000, value=400, key="rep_npts")

    S_grid = np.linspace(float(s_min), float(s_max), int(n_pts))

    if opt_type == "Call":
        payoff_opt = np.maximum(S_grid - K, 0.0)
    else:
        payoff_opt = np.maximum(K - S_grid, 0.0)

    payoff_rep = delta0 * S_grid + B0 * np.exp(r * T)
    err = payoff_rep - payoff_opt

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(S_grid, payoff_opt, label="Option payoff")
    ax1.plot(S_grid, payoff_rep, label="Static replication: Δ·S_T + B·e^{rT}")
    ax1.axvline(S_hedge, linestyle="--")
    ax1.set_xlabel("Underlying at maturity S_T")
    ax1.set_ylabel("Payoff / Terminal value")
    ax1.set_title("Option vs Static Replication (one-shot delta)")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(S_grid, err)
    ax2.axhline(0.0)
    ax2.axvline(S_hedge, linestyle="--")
    ax2.set_xlabel("S_T")
    ax2.set_ylabel("Replication error (Rep - Option)")
    ax2.set_title("Replication error (convexity / gamma effect)")
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("---")

    # ==========================================================
    # Discrete delta-hedging simulation (more “quant”, still simple)
    # ==========================================================
    st.subheader("Discrete delta hedging simulation (BS)")

    with st.expander("Run a discrete hedging simulation (recommended)", expanded=True):

        colh1, colh2 = st.columns(2)
        with colh1:
            n_steps = st.number_input("Re-hedge steps (N)", value=50, min_value=1, max_value=2000, key="rep_N")
            n_paths = st.number_input("Simulation paths", value=2000, min_value=200, max_value=50000, key="rep_paths")
        with colh2:
            seed = st.number_input("Random seed", value=123, min_value=0, max_value=10_000_000, key="rep_seed")
            use_same_delta_spot = st.checkbox("Initial hedge at S₀ (ignore custom hedge spot here)", value=True, key="rep_useS0")

        if st.button("Run hedging simulation", key="rep_run_hedge"):

            try:
                np.random.seed(int(seed))

                N = int(n_steps)
                M = int(n_paths)
                dt = T / N

                # Simulate GBM under risk-neutral dynamics (ignoring q for simplicity)
                Z = np.random.normal(size=(M, N))
                S = np.zeros((M, N + 1))
                S[:, 0] = S0

                for t in range(N):
                    S[:, t+1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

                # Initial option price at t=0 (true model price)
                price_init = float(price0)

                # Initialize hedge portfolio:
                # choose initial spot for delta
                S_init_delta = float(S0) if use_same_delta_spot else float(S_hedge)

                if opt_type == "Call":
                    delta_init = np.array(delta_call(S_init_delta, K, r, sigma, T), ndmin=1).astype(float)[0]
                else:
                    delta_init = np.array(delta_put(S_init_delta, K, r, sigma, T), ndmin=1).astype(float)[0]

                # Portfolio holds delta shares + cash
                cash = price_init - delta_init * S0  # replicate at t=0
                delta_pos = np.full(M, delta_init, dtype=float)

                # Re-hedge through time
                for t in range(N):
                    # cash accrues
                    cash *= np.exp(r * dt)

                    # compute new delta at current time (time to maturity decreases)
                    tau = max(T - (t+1) * dt, 1e-12)
                    S_t = S[:, t+1]

                    if opt_type == "Call":
                        new_delta = np.array(delta_call(S_t, K, r, sigma, tau), dtype=float)
                    else:
                        new_delta = np.array(delta_put(S_t, K, r, sigma, tau), dtype=float)

                    # adjust shares: buy/sell (delta changes)
                    d_delta = new_delta - delta_pos
                    cash -= d_delta * S_t
                    delta_pos = new_delta

                # Terminal portfolio value
                V_T = delta_pos * S[:, -1] + cash

                # Terminal option payoff
                if opt_type == "Call":
                    payoff_T = np.maximum(S[:, -1] - K, 0.0)
                else:
                    payoff_T = np.maximum(K - S[:, -1], 0.0)

                hedge_error = V_T - payoff_T

                st.success("Simulation done")

                cA, cB, cC = st.columns(3)
                cA.metric("Mean hedging error", f"{hedge_error.mean():.6f}")
                cB.metric("Std hedging error", f"{hedge_error.std(ddof=1):.6f}")
                cC.metric("95% quantile |error|", f"{np.quantile(np.abs(hedge_error), 0.95):.6f}")

                # Histogram
                fig3, ax3 = plt.subplots(figsize=(7, 3.8))
                ax3.hist(hedge_error, bins=50)
                ax3.set_title("Discrete hedging error distribution (V_T - payoff)")
                ax3.set_xlabel("Hedging error")
                ax3.set_ylabel("Frequency")
                ax3.grid(True)
                st.pyplot(fig3)

                # Optional download
                df_out = pd.DataFrame({"hedging_error": hedge_error})
                st.download_button(
                    "Download hedging errors (CSV)",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="hedging_errors.csv",
                    mime="text/csv",
                    key="rep_dl_err"
                )

            except Exception as e:
                st.error(f"Hedging simulation error: {e}")