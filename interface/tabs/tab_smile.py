import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pricer.calibration.market_calibrator import MarketSmileCalibrator


def render():
    st.subheader("Volatility Smile")

    ticker_s = st.text_input("Ticker", value="SPY", key="sml_ticker")
    r_s = st.number_input("Risk-free rate r", value=0.04, key="sml_r")

    if st.button("Compute Smile", key="sml_btn"):

        try:
            calibrator = MarketSmileCalibrator(ticker_s, r=r_s)
            df = calibrator.compute_smile()

            if df is None or df.empty:
                st.warning("No valid options found.")
                return

            st.success("Smile computed successfully!")
            st.dataframe(df, use_container_width=True)

            fig = calibrator.plot_smile()
            st.pyplot(fig)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download smile as CSV",
                csv,
                file_name="volatility_smile.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error computing smile: {e}")
