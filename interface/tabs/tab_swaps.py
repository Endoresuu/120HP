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
    st.header("Interest Rate Swap (Fixed vs Floating)")

    col1, col2 = st.columns(2)

    with col1:
        r = st.number_input("Flat interest rate r", value=0.03, key="swap_r")
        T = st.number_input("Maturity T (years)", value=5.0, key="swap_T")
        fixed_rate = st.number_input("Fixed rate K", value=0.03, key="swap_K")

    with col2:
        notional = st.number_input("Notional", value=1.0, key="swap_N")
        freq = st.selectbox("Payment frequency", [1, 2, 4], index=0, key="swap_freq")
        payer = st.radio("Position", ["Payer (pay fixed)", "Receiver (receive fixed)"], key="swap_pos")

    if st.button("Price swap", key="swap_btn"):

        swap = InterestRateSwap(
            fixed_rate=fixed_rate,
            maturity=T,
            freq=freq,
            notional=notional
        )

        par_rate = swap.par_rate(r)
        value = swap.value(r, payer=(payer.startswith("Payer")))

        st.success(f"Par swap rate: {par_rate:.6f}")
        st.info(f"Swap present value: {value:.6f}")