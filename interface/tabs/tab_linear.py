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
    st.header("Forwards & Futures")

    col1, col2 = st.columns(2)

    with col1:
        S0 = st.number_input("Spot S₀", value=100.0, key="lin_S0")
        r = st.number_input("Risk-free rate r", value=0.04, key="lin_r")
        q = st.number_input("Dividend yield q", value=0.0, key="lin_q")
        T = st.number_input("Maturity T (years)", value=1.0, key="lin_T")

    with col2:
        product_type = st.radio(
            "Product type",
            ["Forward", "Future"],
            key="lin_type"
        )
        K = st.number_input(
            "Delivery price K (Forward only)",
            value=100.0,
            disabled=(product_type == "Future"),
            key="lin_K"
        )

    if st.button("Price linear product", key="lin_btn"):

        if product_type == "Forward":
            from pricer.products.forward import Forward
            fwd = Forward(K=K, T=T)
            F0 = fwd.forward_price(S0, r, q)
            V0 = fwd.value(S0, r, q)

            st.success(f"Forward price F₀ = {F0:.4f}")
            st.info(f"Forward value V₀ = {V0:.4f}")

        else:
            from pricer.products.future import Future
            fut = Future(T=T)
            F0 = fut.future_price(S0, r, q)

            st.success(f"Future price = {F0:.4f}")
            st.info("Under constant rates, Future = Forward")