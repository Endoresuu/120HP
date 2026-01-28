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

# --- 1. FONCTIONS CACHÉES (DATA & CALCUL) ---

@st.cache_data(ttl=3600) # Cache de 1h pour les données de marché
def fetch_market_data_cached(ticker):
    """Télécharge le spot et la chaîne d'options."""
    try:
        # On récupère le dernier prix de clôture
        s0_val = float(get_close_price(ticker).iloc[-1])
        # On récupère la chaîne d'options
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
    if opt_type == "Call":
        opt = EuropeanCall(K, T)
    else:
        opt = EuropeanPut(K, T)

    # Logique Volatilité Implicite (Newton)
    if vol_mode == "Implied volatility (Newton)":
        intrinsic = max(S0 - K, 0) if opt_type == "Call" else max(K - S0, 0)

        if market_price < intrinsic:
            res["error"] = f"Market price {market_price:.4f} < intrinsic value {intrinsic:.4f}. Impossible IV."
            return res

        if market_price > S0:
            res["error"] = "Market price cannot exceed spot price."
            return res

        try:
            solver = NewtonImpliedVolSolver()
            # On recrée l'objet MarketOption ici pour éviter les soucis de hashage
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

    # Pricing Black-Scholes avec la volatilité (manuelle ou calculée)
    try:
        market = MarketData(spot=S0, r=r)
        model = BlackScholesModel(market_data=market, sigma=res["sigma"])
        engine = PricingEngine(model=model)

        price = engine.price_european(opt, kind=opt_type.lower())
        res["price"] = price
    except Exception as e:
        res["error"] = f"Pricing error: {e}"

    return res

# --- 2. FONCTION DE RENDU ---

def render():
    st.header("Pricing")

    # ============================
    #       TICKER (Data)
    # ============================
    ticker = st.text_input("Enter a ticker (e.g. SPY, AAPL, MSFT)", key="pr_ticker")

    # Valeurs par défaut
    S0_default = 100.0
    chains = None

    # Récupération des données (Cachée)
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

    # ============================
    #     PARAMETERS (UI)
    # ============================
    col1, col2 = st.columns(2)

    # ----- OPTION SETTINGS -----
    with col1:
        opt_type = st.radio("Option type", ["Call", "Put"], key="pr_opt_type")

        use_atm_strike = st.checkbox("Use ATM strike (K ≈ Spot)", value=False, key="pr_atm")
        if use_atm_strike:
            K = st.number_input("Strike K", value=float(round(S0_default, 2)), disabled=True, key="pr_K_atm")
        else:
            K = st.number_input("Strike K", value=100.0, key="pr_K") # Valeur par défaut fixe pour éviter reset intempestif

        use_auto_maturity = st.checkbox("Use next option expiry from market", value=False, key="pr_auto_T")
        expiry_used = None

        if use_auto_maturity and chains is not None:
            expiry_used, days_auto, T_auto = choose_next_expiry(chains)
            if expiry_used is not None:
                st.info(f"Using next expiry {expiry_used} (~{days_auto} days).")
                T = st.number_input("Maturity T (years)", value=float(round(T_auto, 4)), disabled=True, key="pr_T_auto")
            else:
                st.warning("No future expiry found.")
                T = st.number_input("Maturity T (years)", value=0.5, key="pr_T_manual_fallback")
        else:
            T = st.number_input("Maturity T (years)", value=0.5, key="pr_T")

    # ----- MARKET SETTINGS -----
    with col2:
        use_auto_spot = st.checkbox("Use spot from ticker", value=bool(ticker), key="pr_auto_spot")
        if use_auto_spot:
            S0 = st.number_input("Spot S₀", value=float(round(S0_default, 4)), disabled=True, key="pr_S0_auto")
        else:
            S0 = st.number_input("Spot S₀", value=float(round(S0_default, 4)), key="pr_S0")

        r = st.number_input("Risk-free rate r", value=0.04, key="pr_r")

    # ============================
    #   VOLATILITY METHOD
    # ============================
    st.subheader("Volatility")
    vol_mode = st.radio("Volatility method", ["Manual volatility", "Implied volatility (Newton)"], key="pr_vol_mode")

    sigma_manual = 0.20
    market_price = 0.0

    # Logique d'interface pour le mode Volatilité
    if vol_mode == "Manual volatility":
        sigma_manual = st.number_input("Volatility σ", value=0.20, key="pr_sigma_manual")
    else:
        st.markdown("**Implied volatility (Newton)** — requires a market price.")
        if opt_type == "Put":
            st.warning("Implied vol solver implemented only for CALLs (often).")

        use_auto_market_price = st.checkbox("Use market price from option chain", value=(chains is not None), key="pr_use_chain_price")

        price_auto = None
        if use_auto_market_price and chains is not None:
            expiry_for_price = expiry_used or choose_expiry_closest_to_T(chains, T)[0]
            if expiry_for_price and K > 0:
                price_auto, K_used = get_market_call_price_from_chain(chains, expiry_for_price, K)

            if price_auto is not None:
                st.info(f"Market CALL price {price_auto:.4f} (expiry {expiry_for_price}, strike {K_used})")
                market_price = st.number_input("Market option price", value=float(round(price_auto, 4)), disabled=True, key="pr_market_price_auto")
            else:
                st.warning("No suitable option found — enter price manually.")
                market_price = st.number_input("Market price", value=1.0, key="pr_market_price_manual_fallback")
        else:
            market_price = st.number_input("Market price", value=1.0, key="pr_market_price")

    # ============================
    #       CALCULATE PRICE
    # ============================

    # Bouton d'action
    if st.button("Calculate price", key="pr_btn_calc"):
        # Validation simple avant d'appeler le cache
        errors = []
        if T <= 0: errors.append("Maturity T must be > 0.")
        if K <= 0: errors.append("Strike K must be > 0.")
        if S0 <= 0: errors.append("Spot S₀ must be > 0.")
        if vol_mode == "Implied volatility (Newton)" and market_price <= 0:
            errors.append("Market price must be > 0.")

        if errors:
            for e in errors: st.error(e)
        else:
            # Appel de la fonction cachée
            result = compute_pricing_cached(S0, K, T, r, opt_type, vol_mode, sigma_manual, market_price)
            st.session_state.pricing_result = result

    # Affichage Persistant
    if 'pricing_result' in st.session_state:
        res = st.session_state.pricing_result

        if res.get("error"):
            st.error(res["error"])
        else:
            st.divider()
            if res["implied_vol_computed"]:
                st.success(f"Implied volatility: {res['sigma']:.4f}")

            # Affichage du prix
            if res["price"] is not None:
                st.metric("Option Price", f"{res['price']:.4f}")

            # Détails supplémentaires si besoin
            st.caption(f"Pricing based on: S0={S0:.2f}, K={K:.2f}, T={T:.4f}, r={r}, σ={res['sigma']:.4f}")
