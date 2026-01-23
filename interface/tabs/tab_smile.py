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
    st.subheader("Volatility Smile")

    ticker_s = st.text_input("Ticker", value="SPY", key="sml_ticker")
    r_s = st.number_input("Risk-free rate r", value=0.04, key="sml_r")

    if st.button("Compute Smile", key="sml_btn"):

        try:
            calibrator = MarketSmileCalibrator(ticker_s, r=r_s)
            df = calibrator.compute_smile()

            if df.empty:
                st.warning("No valid options found.")
            else:
                st.success("Smile computed successfully!")
                st.dataframe(df)

                fig = calibrator.plot_smile()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error computing smile: {e}")