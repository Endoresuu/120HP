import os
import sys

# Ajouter la racine du projet (/workspaces/120HP) au sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from pricer.market.import_data import get_close_price
from pricer.market.data import MarketData
from pricer.models.black_scholes import BlackScholesModel
from pricer.products.vanilla import EuropeanCall, EuropeanPut
from pricer.pricing.engine import PricingEngine
from pricer.calibration.implied_vol import NewtonImpliedVolSolver


st.set_page_config(page_title="Option Pricer", page_icon="💰")

st.markdown(
    "<h1 style='text-align:center;'> Option Pricer 💰</h1>",
    unsafe_allow_html=True
)

# ============================
#        TICKER
# ============================
st.header("Ticker")

ticker = st.text_input("Enter a ticker (e.g. SPY, AAPL, MSFT)")

if ticker:
    try:
        S0_default = float(get_close_price(ticker).iloc[-1])
        st.success(f"Current Spot: {S0_default}")
    except:
        st.error("Invalid ticker")
        S0_default = 100.0
else:
    S0_default = 100.0


# ============================
#      PARAMETERS
# ============================
col1, col2 = st.columns(2)

with col1:
    opt_type = st.radio("Option type", ["Call", "Put"])
    K = st.number_input("Strike K")
    T = st.number_input("Maturity T (years)")

with col2:
    S0 = st.number_input("Spot", value=S0_default)
    r = st.number_input("Risk-free rate r", value=0.04)

# ============================
#   CHOIX DE LA VOLATILITÉ
# ============================
vol_mode = st.radio(
    "Volatility method",
    ["Manual volatility", "Implied volatility (Newton)"]
)

if vol_mode == "Manual volatility":
    sigma = st.number_input("Volatility sigma", value=0.20)
else:
    market_price = st.number_input("Market price of the option (for implied vol)", value=1.0)
    sigma = None  # Calculé plus tard


# ============================
#       PRICE BUTTON
# ============================
if st.button("Calculate price"):

    # --- Option
    if opt_type == "Call":
        opt = EuropeanCall(K=K, T=T)
    else:
        opt = EuropeanPut(K=K, T=T)

    # --- Détermination de la volatilité
    if vol_mode == "Implied volatility (Newton)":
        solver = NewtonImpliedVolSolver()

        # construire l'objet demandé par le solver
        from pricer.products.market_option import MarketOption as SolverOption
        opt_solver = SolverOption(
            S0=S0,
            K=K,
            T=T,
            r=r,
            price_mkt=market_price
        )

        sigma = solver.solve(opt_solver)

        if sigma is None or sigma != sigma:  # nan safety
            st.error("Could not compute implied volatility.")
            st.stop()

        st.success(f"Implied volatility (Newton): {sigma:.4f}")

    # --- Market data
    market = MarketData(spot=S0, r=r, q=0.0)

    # --- Model
    model = BlackScholesModel(market_data=market, sigma=sigma)

    # --- Pricing
    engine = PricingEngine(model=model)
    price = engine.price_european(opt, kind=opt_type.lower())

    st.success(f"Option price: {price:.4f}")
