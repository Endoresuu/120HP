import streamlit as st
import numpy as np
import pandas as pd

from pricer.market.import_data import get_option_chain, get_close_price
from pricer.market.data import MarketData
from pricer.models.black_scholes import BlackScholesModel
from pricer.pricing.engine import PricingEngine
from pricer.products.vanilla import EuropeanCall, EuropeanPut
from pricer.calibration.implied_vol import NewtonImpliedVolSolver
from pricer.products.market_option import MarketOption
from interface.utils.market_helpers import (
    choose_next_expiry,
    choose_expiry_closest_to_T,
    get_market_call_price_from_chain
)

# ======================================================
# 1. FONCTIONS CACHÉES (DATA & CALCUL)
# ======================================================

@st.cache_data(ttl=3600)  # Cache de 1h pour les données de marché
def fetch_market_data_cached(ticker):
    """Télécharge le spot et la chaîne d'options."""
    try:
        s0_val = float(get_close_price(ticker).iloc[-1])
        chains_data = get_option_chain(ticker)
        return s0_val, chains_data, None
    except Exception as e:
        return 100.0, None, str(e)


@st.cache_data
def compute_pricing_cached(S0, K, T, r, opt_type, vol_mode, manual_sigma, market_price):
    """
    Effectue le calcul de pricing (et de volatilité implicite si nécessaire).
    Retourne un dictionnaire de résultats.
    """
    res = {
        "price": None,
        "sigma": manual_sigma,
        "implied_vol_computed": False,
        "error": None
    }

    # Création de l'option
    opt = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

    # Logique Volatilité Implicite (Newton)
    if vol_mode == "Implied volatility (Newton)":
        intrinsic = max(S0 - K, 0) if opt_type == "Call" else max(K - S0, 0)

        if market_price < intrinsic:
            res["error"] = (
                f"Market price {market_price:.4f} < intrinsic value "
                f"{intrinsic:.4f}. Impossible IV."
            )
            return res

        try:
            solver = NewtonImpliedVolSolver()
            opt_solver = MarketOption(S0=S0, K=K, T=T, r=r, price_mkt=market_price)
            sigma_calc = solver.solve(opt_solver)

            if sigma_calc is None or not np.isfinite(sigma_calc):
                res["error"] = "Implied volatility could not be computed (solver failed)."
                return res

            res["sigma"] = sigma_calc
            res["implied_vol_computed"] = True

        except Exception as e:
            res["error"] = f"Solver error: {e}"
            return res

    # Pricing Black-Scholes
    try:
        market = MarketData(spot=S0, r=r)
        model = BlackScholesModel(market_data=market, sigma=res["sigma"])
        engine = PricingEngine(model=model)
        res["price"] = engine.price_european(opt, kind=opt_type.lower())
    except Exception as e:
        res["error"] = f"Pricing error: {e}"

    return res


# ======================================================
# 2. FONCTION DE RENDU
# ======================================================

def render():
    st.header("Pricing")
    st.caption("Black–Scholes pricing with market data and implied volatility")

    # ============================
    #       TICKER (Data)
    # ============================
    ticker = st.text_input(
        "Enter a ticker (e.g. SPY, AAPL, MSFT)",
        key="pr_ticker"
    )

    S0_default = 100.0
    chains = None

    if ticker:
        with st.spinner("Fetching market data..."):
            fetched_S0, fetched_chains, error_msg = fetch_market_data_cached(ticker)
            if error_msg:
                st.error(f"Error fetching data: {error_msg}")
            else:
                S0_default = fetched_S0
                chains = fetched_chains
                st.success(f"Current Spot from market: {S0_default:.4f}")
    else:
        st.info("Enter a ticker to enable automatic market features.")

    st.divider()

    # ============================
    #       MAIN LAYOUT
    # ============================
    col_inputs, col_outputs = st.columns([1.2, 1.8], gap="large")

    # ============================
    #       INPUTS
    # ============================
    with col_inputs:

        # ----- OPTION SETTINGS -----
        st.subheader("Contract")

        opt_type = st.radio("Option type", ["Call", "Put"], key="pr_opt_type", horizontal=True)

        use_atm_strike = st.checkbox("Use ATM strike (K ≈ Spot)", key="pr_atm")
        if use_atm_strike:
            K = st.number_input("Strike K", value=S0_default, disabled=True, key="pr_K_atm")
        else:
            K = st.number_input("Strike K", value=100.0, key="pr_K")

        use_auto_maturity = st.checkbox("Use next option expiry from market", key="pr_auto_T")
        expiry_used = None

        if use_auto_maturity and chains is not None:
            expiry_used, days_auto, T_auto = choose_next_expiry(chains)
            if expiry_used:
                st.info(f"Using next expiry {expiry_used} (~{days_auto} days).")
                T = st.number_input("Maturity T (years)", value=T_auto, disabled=True)
            else:
                T = st.number_input("Maturity T (years)", value=0.5)
        else:
            T = st.number_input("Maturity T (years)", value=0.5)

        # ----- MARKET SETTINGS -----
        st.subheader("Market")

        use_auto_spot = st.checkbox("Use spot from ticker", value=bool(ticker))
        if use_auto_spot:
            S0 = st.number_input("Spot S₀", value=S0_default, disabled=True)
        else:
            S0 = st.number_input("Spot S₀", value=S0_default)

        r = st.number_input("Risk-free rate r", value=0.04)

        # ============================
        #   VOLATILITY METHOD
        # ============================
        st.subheader("Volatility")

        vol_mode = st.radio(
            "Volatility method",
            ["Manual volatility", "Implied volatility (Newton)"],
            key="pr_vol_mode"
        )

        sigma_manual = 0.20
        market_price = 0.0

        if vol_mode == "Manual volatility":
            sigma_manual = st.number_input("Volatility σ", value=0.20)
        else:
            st.markdown("**Implied volatility (Newton)** — requires a market price.")

            use_auto_market_price = st.checkbox(
                "Use market price from option chain",
                value=(chains is not None)
            )

            if use_auto_market_price and chains is not None:
                expiry_for_price = expiry_used or choose_expiry_closest_to_T(chains, T)[0]
                price_auto, K_used = get_market_call_price_from_chain(
                    chains, expiry_for_price, K
                )

                if price_auto is not None:
                    st.info(
                        f"Market CALL price {price_auto:.4f} "
                        f"(expiry {expiry_for_price}, strike {K_used})"
                    )
                    market_price = st.number_input(
                        "Market option price",
                        value=price_auto,
                        disabled=True
                    )
                else:
                    market_price = st.number_input("Market price", value=1.0)
            else:
                market_price = st.number_input("Market price", value=1.0)

        # ============================
        #       ACTION
        # ============================
        calc = st.button("Calculate price", use_container_width=True)

    # ============================
    #       OUTPUTS
    # ============================
    with col_outputs:

        st.subheader("Results")

        if calc:
            errors = []
            if T <= 0: errors.append("Maturity T must be > 0.")
            if K <= 0: errors.append("Strike K must be > 0.")
            if S0 <= 0: errors.append("Spot S₀ must be > 0.")
            if vol_mode == "Implied volatility (Newton)" and market_price <= 0:
                errors.append("Market price must be > 0.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.session_state.pricing_result = compute_pricing_cached(
                    S0, K, T, r, opt_type, vol_mode, sigma_manual, market_price
                )

        if "pricing_result" in st.session_state:
            res = st.session_state.pricing_result

            if res.get("error"):
                st.error(res["error"])
            else:
                if res["implied_vol_computed"]:
                    st.success(f"Implied volatility: {res['sigma']:.4f}")

                if res["price"] is not None:
                    st.metric("Option Price", f"{res['price']:.4f}")

                st.caption(
                    f"S0={S0:.2f}, K={K:.2f}, T={T:.4f}, r={r}, σ={res['sigma']:.4f}"
                )
