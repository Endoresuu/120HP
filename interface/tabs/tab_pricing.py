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

from interface.utils.market_helpers import (
    choose_next_expiry,
    choose_expiry_closest_to_T,
    get_market_call_price_from_chain
)

def render():

    st.header("Pricing")

    # ============================
    #       TICKER
    # ============================
    ticker = st.text_input("Enter a ticker (e.g. SPY, AAPL, MSFT)", key="pr_ticker")
    S0_default = 100.0
    chains = None

    if ticker:
        try:
            S0_default = float(get_close_price(ticker).iloc[-1])
            st.success(f"Current Spot from market: {S0_default:.4f}")
            chains = get_option_chain(ticker)
        except Exception as e:
            st.error(f"Invalid ticker or data error: {e}")
            S0_default = 100.0
    else:
        st.info("Enter a ticker to enable automatic market features.")

    # ============================
    #     PARAMETERS
    # ============================
    col1, col2 = st.columns(2)

    # ----- OPTION SETTINGS -----
    with col1:
        opt_type = st.radio("Option type", ["Call", "Put"], key="pr_opt_type")

        use_atm_strike = st.checkbox("Use ATM strike (K ≈ Spot)", value=False, key="pr_atm")
        if use_atm_strike:
            K = st.number_input("Strike K", value=float(round(S0_default, 2)),
                                disabled=True, key="pr_K_atm")
        else:
            K = st.number_input("Strike K", value=0.0, key="pr_K")

        use_auto_maturity = st.checkbox("Use next option expiry from market", value=False, key="pr_auto_T")
        expiry_used = None

        if use_auto_maturity and chains is not None:
            expiry_used, days_auto, T_auto = choose_next_expiry(chains)
            if expiry_used is not None:
                st.info(f"Using next expiry {expiry_used} (~{days_auto} days).")
                T = st.number_input("Maturity T (years)", value=float(round(T_auto, 4)),
                                    disabled=True, key="pr_T_auto")
            else:
                st.warning("No future expiry found.")
                T = st.number_input("Maturity T (years)", value=0.5, key="pr_T_manual_fallback")
        else:
            T = st.number_input("Maturity T (years)", value=0.5, key="pr_T")

    # ----- MARKET SETTINGS -----
    with col2:
        use_auto_spot = st.checkbox("Use spot from ticker", value=bool(ticker), key="pr_auto_spot")
        if use_auto_spot:
            S0 = st.number_input("Spot S₀", value=float(round(S0_default, 4)),
                                 disabled=True, key="pr_S0_auto")
        else:
            S0 = st.number_input("Spot S₀", value=float(round(S0_default, 4)),
                                 key="pr_S0")

        r = st.number_input("Risk-free rate r", value=0.04, key="pr_r")

    # ============================
    #   VOLATILITY METHOD
    # ============================
    st.subheader("Volatility")

    vol_mode = st.radio(
        "Volatility method",
        ["Manual volatility", "Implied volatility (Newton)"],
        key="pr_vol_mode"
    )

    sigma = None
    market_price = None
    use_auto_market_price = False

    if vol_mode == "Manual volatility":
        sigma = st.number_input("Volatility σ", value=0.20, key="pr_sigma_manual")

    else:
        st.markdown("**Implied volatility (Newton)** — requires a market price.")
        if opt_type == "Put":
            st.warning("Implied vol solver implemented only for CALLs.")

        use_auto_market_price = st.checkbox(
            "Use market price from option chain",
            value=(chains is not None),
            key="pr_use_chain_price"
        )

        if use_auto_market_price and chains is not None:

            expiry_for_price = expiry_used or choose_expiry_closest_to_T(chains, T)[0]

            price_auto = None
            if expiry_for_price and K > 0:
                price_auto, K_used = get_market_call_price_from_chain(
                    chains, expiry_for_price, K
                )

            if price_auto is not None:
                st.info(f"Market CALL price {price_auto:.4f} (expiry {expiry_for_price}, strike {K_used})")
                market_price = st.number_input(
                    "Market option price",
                    value=float(round(price_auto, 4)),
                    disabled=True,
                    key="pr_market_price_auto"
                )
            else:
                st.warning("No suitable option found — enter price manually.")
                use_auto_market_price = False

        if not use_auto_market_price:
            market_price = st.number_input("Market price", value=1.0, key="pr_market_price")

    # ============================
    #       CALCULATE PRICE
    # ============================
    if st.button("Calculate price", key="pr_btn_calc"):

        errors = []
        if T <= 0:
            errors.append("Maturity T must be > 0.")
        if K <= 0:
            errors.append("Strike K must be > 0.")
        if S0 <= 0:
            errors.append("Spot S₀ must be > 0.")
        if vol_mode == "Implied volatility (Newton)" and (
            market_price is None or market_price <= 0):
            errors.append("Market price must be > 0.")

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        opt = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

        if vol_mode == "Implied volatility (Newton)":

            intrinsic = max(S0 - K, 0) if opt_type == "Call" else max(K - S0, 0)

            if market_price < intrinsic:
                st.error(f"Market price {market_price} < intrinsic value {intrinsic}. Impossible IV.")
                st.stop()

            if market_price > S0:
                st.error("Market price cannot exceed spot price.")
                st.stop()

            solver = NewtonImpliedVolSolver()
            opt_solver = MarketOption(S0=S0, K=K, T=T, r=r, price_mkt=market_price)
            sigma = solver.solve(opt_solver)

            if sigma is None or not np.isfinite(sigma):
                st.error("Implied volatility could not be computed.")
                st.stop()

            st.success(f"Implied volatility: {sigma:.4f}")

        market = MarketData(spot=S0, r=r)
        model = BlackScholesModel(market_data=market, sigma=sigma)
        engine = PricingEngine(model=model)

        price = engine.price_european(opt, kind=opt_type.lower())
        st.success(f"Option price: {price:.4f}")

