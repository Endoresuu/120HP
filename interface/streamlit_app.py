import streamlit as st
from pricer.market.data import MarketData
from pricer.market.import_data import get_close_price
from pricer.models.black_scholes import BlackScholesModel
from pricer.models.heston import HestonModel
from pricer.pricing.engine import PricingEngine
from pricer.products.vanilla import EuropeanCall, EuropeanPut
from pricer.calibration.market_calibrator import MarketSmileCalibrator
from pricer.calibration.surface_calibrator import Calibrator
import numpy as np
import pandas as pd


import os
import sys

# Trouve la racine du projet
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ajoute au PYTHONPATH
sys.path.insert(0, ROOT_DIR)

print(">>> USING ROOT_DIR =", ROOT_DIR)
# -------------------------
#   PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Option Pricer", page_icon="💰", layout="wide")

st.markdown("<h1 style='text-align:center;'>Option Pricer 💰</h1>", unsafe_allow_html=True)

# -------------------------
#   TABS
# -------------------------
tab_price, tab_smile, tab_surface, tab_heston, tab_greeks, tab_calib = st.tabs(
    ["Pricer", "Volatility Smile", "Volatility Surface", "Heston Monte Carlo", "Greeks", "Calibration BS/Heston"]
)

with tab_price:

    ticker = st.text_input("Ticker", value="SPY", key="prc_ticker")

    if ticker:
        try:
            S0_default = float(get_close_price(ticker).iloc[-1])
            st.success(f"Current spot: {S0_default}")
        except:
            st.error("Invalid ticker")
            S0_default = 100.0

    opt_type = st.radio("Option type", ["Call", "Put"], key="prc_opt_type")

    col1, col2 = st.columns(2)

    with col1:
        K = st.number_input("Strike K", key="prc_K")
        T = st.number_input("Maturity T (years)", key="prc_T")

    with col2:
        S0 = st.number_input("Spot", value=S0_default, key="prc_S0")
        r = st.number_input("Risk-free rate r", value=0.04, key="prc_r")

    vol_method = st.radio("Volatility method", ["Manual", "Implied (Newton)"], key="prc_vol_method")

    if vol_method == "Manual":
        sigma = st.number_input("Sigma", 0.20, key="prc_sigma")

    else:
        market_price = st.number_input("Market price (for IV)", key="prc_mkt_price")
        sigma = None  # sera calculée plus bas

    if st.button("Calculate price", key="prc_btn"):

        market = MarketData(spot=S0, r=r, q=0.0)

        # Si volatilité implicite demandée
        if vol_method == "Implied (Newton)":
            try:
                solver = NewtonImpliedVolSolver()
                opt_tmp = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)
                sigma = solver.solve_market_price(opt_tmp, market, market_price)
                st.info(f"Implied Volatility: {sigma:.4f}")
            except:
                st.error("Could not compute implied volatility.")
                st.stop()

        model = BlackScholesModel(market_data=market, sigma=sigma)
        opt = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)
        engine = PricingEngine(model)

        price = engine.price_european(opt, kind=opt_type.lower())
        st.success(f"BS Price: {price:.4f}")

with tab_heston:

    st.subheader("Heston Monte Carlo Simulation")

    ticker_h = st.text_input("Ticker", value="SPY", key="hst_ticker")

    S0_h = st.number_input("Spot", key="hst_S0")
    r_h = st.number_input("Rate r", value=0.04, key="hst_r")
    T_h = st.number_input("Maturity T", key="hst_T")

    n_paths = st.number_input("Number of paths", value=5000, key="hst_paths")
    n_steps = st.number_input("Steps", value=200, key="hst_steps")

    kappa = st.number_input("Kappa", value=1.2, key="hst_kappa")
    theta = st.number_input("Theta", value=0.04, key="hst_theta")
    v0 = st.number_input("v0", value=0.04, key="hst_v0")
    sigma_v = st.number_input("Sigma v", value=0.3, key="hst_sigma_v")
    rho = st.number_input("Rho correlation", value=-0.6, key="hst_rho")

    opt_type_h = st.radio("Option type", ["Call", "Put"], key="hst_opt_type")
    K_h = st.number_input("Strike", key="hst_K")

    if st.button("Simulate Heston MC", key="hst_run"):

        market = MarketData(
            spot=S0_h,
            r=r_h,
            T=T_h,
            v_0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            n_paths=n_paths,
            n_steps=n_steps,
        )

        model = HestonModel()
        opt = EuropeanCall(K_h, T_h) if opt_type_h == "Call" else EuropeanPut(K_h, T_h)

        price = model.price_european(opt, market)
        st.success(f"Heston MC price: {price:.4f}")

with tab_smile:

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

                fig = calibrator.plot_smile(return_fig=True)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error computing smile: {e}")


with tab_surface:

    st.subheader("Volatility Surface (2D)")

    ticker_sf = st.text_input("Ticker for surface", value="SPY", key="surf_ticker")
    n_mat = st.slider("Number of maturities to use", 2, 20, 5, key="surf_mat")

    if st.button("Compute Volatility Surface", key="surf_btn"):

        try:
            # ===== 1. Load data =====
            chains = get_option_chain(ticker_sf)
            S0 = float(get_close_price(ticker_sf).iloc[-1])
            r = 0.04   # tu peux ajouter un input si tu veux

            # ===== 2. Select maturities =====
            expiries = sorted(chains.keys())[:n_mat]
            maturities = []

            for e in expiries:
                T = (datetime.strptime(e, "%Y-%m-%d") - datetime.today()).days / 365
                maturities.append(T)

            # ===== 3. Get strikes union =====
            all_strikes = sorted(set().union(*[set(chains[e]["strike"]) for e in expiries]))
            strikes = np.array(all_strikes)

            # ===== 4. Build price matrix (maturities x strikes) =====
            price_matrix = np.zeros((len(maturities), len(strikes)))

            for i, e in enumerate(expiries):
                df = chains[e].set_index("strike")

                for j, K in enumerate(strikes):
                    if K in df.index:
                        price_matrix[i, j] = df.loc[K]["lastPrice"]
                    else:
                        price_matrix[i, j] = np.nan   # strike non coté

            # ===== 5. Remove columns fully NaN =====
            valid_cols = ~np.isnan(price_matrix).all(axis=0)
            strikes = strikes[valid_cols]
            price_matrix = price_matrix[:, valid_cols]

            # ===== 6. Calibrate surface =====
            cal = Calibrator(
                strikes=strikes,
                maturities=maturities,
                S0=S0,
                r=r,
                price_matrix=price_matrix
            )

            vol_surface = cal.build_surface()

            st.success("Volatility surface computed!")

            st.write("Strikes:", strikes)
            st.write("Maturities:", maturities)

            # Surface plot
            fig = plt.figure(figsize=(8,5))
            plt.imshow(vol_surface.surface, aspect='auto', cmap='viridis',
                        extent=[strikes.min(), strikes.max(), maturities[-1], maturities[0]])
            plt.colorbar(label="Volatility")
            plt.xlabel("Strike")
            plt.ylabel("Maturity")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error while computing volatility surface: {e}")

with tab_calib:

    st.subheader("Model Calibration (BS / Heston)")

    ticker_c = st.text_input("Ticker", value="SPY", key="cal_ticker")
    method_c = st.radio("Model to calibrate", ["Black-Scholes", "Heston"], key="cal_model")

    if st.button("Run Calibration", key="cal_btn"):

        try:
            if method_c == "Black-Scholes":
                cal = MarketSmileCalibrator(ticker_c)
                df = cal.compute_smile()
                st.dataframe(df)
                st.success("BS calibration finished!")

            else:
                st.info("Heston calibration requires implementation.")
                # Je peux te coder une vraie calibration Heston si tu veux.

        except Exception as e:
            st.error(f"Calibration error: {e}")

with tab_greeks:

    st.subheader("Greeks (Black–Scholes)")

    col1, col2 = st.columns(2)

    with col1:
        S0_g = st.number_input("Spot", value=100.0, key="grk_S0")
        K_g = st.number_input("Strike", value=100.0, key="grk_K")
        T_g = st.number_input("Maturity", value=1.0, key="grk_T")

    with col2:
        r_g = st.number_input("Rate", value=0.04, key="grk_r")
        sigma_g = st.number_input("Volatility", value=0.20, key="grk_sigma")
        opt_type_g = st.radio("Option type", ["Call", "Put"], key="grk_type")

    if st.button("Compute Greeks", key="grk_btn"):

        try:
            market = MarketData(spot=S0_g, r=r_g, q=0.0)
            model = BlackScholesModel(market_data=market, sigma=sigma_g)
            opt = EuropeanCall(K_g, T_g) if opt_type_g == "Call" else EuropeanPut(K_g, T_g)

            greeks = model.greeks(opt)

            st.success("Greeks computed!")
            st.json(greeks)

        except Exception as e:
            st.error(f"Error computing Greeks: {e}")
