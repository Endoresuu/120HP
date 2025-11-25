import streamlit as st

from pricer.market.import_data import get_close_price
from pricer.market.data import MarketData
from pricer.models.black_scholes import BlackScholesModel
from pricer.products.vanilla import EuropeanCall, EuropeanPut
from pricer.pricing.engine import PricingEngine


st.set_page_config(page_title="Option Pricer", page_icon="💰")

st.markdown(
    "<h1 style='text-align:center;'> Option Pricer 💰</h1>",
    unsafe_allow_html=True
)

# ============================
#       TICKER
# ============================
st.header("Ticker")

ticker = st.text_input("Enter a ticker (e.g. SPY, AAPL, MSFT)")

if ticker:
    try:
        S0_default = float(get_close_price(ticker).iloc[-1])
        st.success(f"Spot actuel : {S0_default}")
    except:
        st.error("Ticker invalide")
        S0_default = 100.0
else:
    S0_default = 100.0


# ============================
#     PARAMETERS
# ============================
col1, col2 = st.columns(2)

with col1:
    opt_type = st.radio("Option type", ["Call", "Put"])
    K = st.number_input("Strike K", value=100.0)
    T = st.number_input("Maturity T (years)", value=1.0)

with col2:
    S0 = st.number_input("Spot S0", value=S0_default)
    r = st.number_input("Risk-free rate r", value=0.04)
    sigma = st.number_input("Volatility sigma", value=0.20)


# ============================
#     PRICE BUTTON
# ============================
if st.button("Calculate price"):

    # --- Market data
    market = MarketData(spot=S0, r=r, q=0.0)

    # --- Model
    model = BlackScholesModel(market_data=market, sigma=sigma)

    # --- Option
    if opt_type == "Call":
        opt = EuropeanCall(K=K, T=T)
    else:
        opt = EuropeanPut(K=K, T=T)

    # --- Pricing
    engine = PricingEngine(model=model)
    price = engine.price_european(opt, kind=opt_type.lower())

    st.success(f"Prix de l'option : {price:.4f}")
