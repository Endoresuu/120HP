import streamlit as st
from pricer.market.data import MarketData
from pricer.models.black_scholes import BlackScholesModel
from pricer.models.heston import HestonModel
from pricer.pricing.engine import PricingEngine
from pricer.products.vanilla import EuropeanCall, EuropeanPut
from pricer.calibration.market_calibrator import MarketSmileCalibrator
from pricer.calibration.surface_calibrator import Calibrator
from pricer.market.import_data import get_option_chain, get_close_price
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour le 3D


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

    st.header("Pricing")

    # ============================
    #       TICKER
    # ============================
    ticker = st.text_input("Enter a ticker (e.g. SPY, AAPL, MSFT)")
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
        opt_type = st.radio("Option type", ["Call", "Put"])

        # Strike
        use_atm_strike = st.checkbox("Use ATM strike (K ≈ Spot)", value=False)
        if use_atm_strike:
            K = st.number_input("Strike K", value=float(round(S0_default, 2)), disabled=True)
        else:
            K = st.number_input("Strike K", value=0.0)

        # Maturity
        use_auto_maturity = st.checkbox("Use next option expiry from market", value=False)
        expiry_used = None

        if use_auto_maturity and chains is not None:
            expiry_used, days_auto, T_auto = choose_next_expiry(chains)
            if expiry_used is not None:
                st.info(f"Using next expiry {expiry_used} (~{days_auto} days).")
                T = st.number_input("Maturity T (years)", value=float(round(T_auto, 4)), disabled=True)
            else:
                st.warning("No future expiry found.")
                T = st.number_input("Maturity T (years)", value=0.5)
        else:
            T = st.number_input("Maturity T (years)", value=0.5)

    # ----- MARKET SETTINGS -----
    with col2:
        use_auto_spot = st.checkbox("Use spot from ticker", value=bool(ticker))
        if use_auto_spot:
            S0 = st.number_input("Spot S₀", value=float(round(S0_default, 4)), disabled=True)
        else:
            S0 = st.number_input("Spot S₀", value=float(round(S0_default, 4)))

        r = st.number_input("Risk-free rate r", value=0.04)

    # ============================
    #   VOLATILITY METHOD
    # ============================
    st.subheader("Volatility")

    vol_mode = st.radio(
        "Volatility method",
        ["Manual volatility", "Implied volatility (Newton)"]
    )

    sigma = None
    market_price = None
    use_auto_market_price = False

    if vol_mode == "Manual volatility":
        sigma = st.number_input("Volatility σ", value=0.20)

    else:
        st.markdown("**Implied volatility (Newton)** — requires a market price.")

        if opt_type == "Put":
            st.warning("Implied vol solver implemented only for CALLs.")

        use_auto_market_price = st.checkbox(
            "Use market price from option chain",
            value=(chains is not None)
        )

        if use_auto_market_price and chains is not None:

            expiry_for_price = expiry_used
            if expiry_for_price is None:
                expiry_for_price, _, _ = choose_expiry_closest_to_T(chains, T)

            price_auto = None
            if expiry_for_price and K > 0:
                price_auto, K_used = get_market_call_price_from_chain(chains, expiry_for_price, K)

            if price_auto is not None:
                st.info(f"Market CALL price {price_auto:.4f} (expiry {expiry_for_price}, strike {K_used})")
                market_price = st.number_input(
                    "Market option price",
                    value=float(round(price_auto, 4)),
                    disabled=True
                )
            else:
                st.warning("No suitable option found — enter price manually.")
                use_auto_market_price = False

        if not use_auto_market_price:
            market_price = st.number_input("Market price", value=1.0)

    # ============================
    #       CALCULATE PRICE
    # ============================
    if st.button("Calculate price"):

        # ---- Validation ----
        errors = []
        if T <= 0:
            errors.append("Maturity T must be > 0.")
        if K <= 0:
            errors.append("Strike K must be > 0.")
        if S0 <= 0:
            errors.append("Spot S₀ must be > 0.")
        if vol_mode == "Implied volatility (Newton)" and market_price <= 0:
            errors.append("Market price must be > 0.")

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        # ---- Build the option ----
        opt = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

        # ---- Implied Vol ----
        if vol_mode == "Implied volatility (Newton)":

            intrinsic = max(S0 - K, 0) if opt_type == "Call" else max(K - S0, 0)

            if market_price < intrinsic:
                st.error(f"Market price {market_price} < intrinsic value {intrinsic}. Impossible IV.")
                st.stop()

            if market_price > S0:
                st.error("Market price cannot exceed spot price.")
                st.stop()

            solver = NewtonImpliedVolSolver()
            opt_solver = SolverOption(S0=S0, K=K, T=T, r=r, price_mkt=market_price)
            sigma = solver.solve(opt_solver)

            if sigma is None or not np.isfinite(sigma):
                st.error("Implied volatility could not be computed.")
                st.stop()

            st.success(f"Implied volatility: {sigma:.4f}")

        # ---- Pricing ----
        market = MarketData(spot=S0, r=r)
        model = BlackScholesModel(market_data=market, sigma=sigma)
        engine = PricingEngine(model=model)

        price = engine.price_european(opt, kind=opt_type.lower())

        st.success(f"Option price: {price:.4f}")



# with tab_price:

    # ticker = st.text_input("Ticker", value="SPY", key="prc_ticker")

    # if ticker:
    #     try:
    #         S0_default = float(get_close_price(ticker).iloc[-1])
    #         st.success(f"Current spot: {S0_default}")
    #     except:
    #         st.error("Invalid ticker")
    #         S0_default = 100.0

    # opt_type = st.radio("Option type", ["Call", "Put"], key="prc_opt_type")

    # col1, col2 = st.columns(2)

    # with col1:
    #     K = st.number_input("Strike K", key="prc_K")
    #     T = st.number_input("Maturity T (years)", key="prc_T")

    # with col2:
    #     S0 = st.number_input("Spot", value=S0_default, key="prc_S0")
    #     r = st.number_input("Risk-free rate r", value=0.04, key="prc_r")

    # vol_method = st.radio("Volatility method", ["Manual", "Implied (Newton)"], key="prc_vol_method")

    # if vol_method == "Manual":
    #     sigma = st.number_input("Sigma", 0.20, key="prc_sigma")

    # else:
    #     market_price = st.number_input("Market price (for IV)", key="prc_mkt_price")
    #     sigma = None  # sera calculée plus bas

    # if st.button("Calculate price", key="prc_btn"):

    #     market = MarketData(spot=S0, r=r, q=0.0)

    #     # Si volatilité implicite demandée
    #     if vol_method == "Implied (Newton)":
    #         try:
    #             solver = NewtonImpliedVolSolver()
    #             opt_tmp = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)
    #             sigma = solver.solve_market_price(opt_tmp, market, market_price)
    #             st.info(f"Implied Volatility: {sigma:.4f}")
    #         except:
    #             st.error("Could not compute implied volatility.")
    #             st.stop()

    #     model = BlackScholesModel(market_data=market, sigma=sigma)
    #     opt = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)
    #     engine = PricingEngine(model)

    #     price = engine.price_european(opt, kind=opt_type.lower())
    #     st.success(f"BS Price: {price:.4f}")

with tab_heston:

    st.subheader("Heston Monte Carlo")

    # ===========================
    # Paramètres du modèle
    # ===========================
    col1, col2 = st.columns(2)

    with col1:
        S0 = st.number_input("Spot S₀", value=100.0, key="hst_S0")
        r = st.number_input("Risk-free rate r", value=0.04, key="hst_r")
        T = st.number_input("Maturity T (years)", value=1.0, key="hst_T")
        K = st.number_input("Strike K", value=100.0, key="hst_K")
        opt_type = st.radio("Option type", ["Call", "Put"], key="hst_opt_type")

    with col2:
        v0 = st.number_input("Initial variance v₀", value=0.04, key="hst_v0")
        kappa = st.number_input("Mean reversion κ", value=1.5, key="hst_kappa")
        theta = st.number_input("Long-run variance θ", value=0.04, key="hst_theta")
        sigma_v = st.number_input("Vol of variance σᵥ", value=0.3, key="hst_sigma_v")
        rho = st.number_input("Correlation ρ", value=-0.7, key="hst_rho")
        n_steps = st.number_input("Time steps", value=100, key="hst_steps")
        n_paths = st.number_input("MC paths", value=5000, key="hst_paths")

    if st.button("Run Heston simulation", key="hst_run"):

        try:
            # On crée MarketData avec les infos de base
            market = MarketData(spot=S0, r=r, T=T)

            # Paramètres Heston
            market.v_0 = v0
            market.kappa = kappa
            market.theta = theta
            market.sigma_v = sigma_v
            market.rho = rho
            market.n_steps = int(n_steps)
            market.n_paths = int(n_paths)

            # Option
            if opt_type == "Call":
                option = EuropeanCall(K, T)
            else:
                option = EuropeanPut(K, T)

            # Modèle Heston
            heston = HestonModel()
            S_paths, v_paths = heston.simulate_paths(market)

            # Prix
            price_heston = heston.price_european(option, market)
            st.success(f"Heston MC price = {price_heston:.4f}")

            # (Trajectoires / histogramme etc. si tu as gardé le reste du code)

        except Exception as e:
            st.error(f"Error in Heston Monte Carlo: {e}")


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

    interp_method = st.selectbox(
        "Interpolation method",
        ["None", "Bilinear (2D interpolation)"],
        key="surf_interp_method"
    )

    show_3d = st.checkbox("Show 3D surface", value=False, key="surf_show_3d")

    if st.button("Compute Volatility Surface", key="surf_btn"):

        try:
            # =========================
            # 1) Market data
            # =========================
            chains = get_option_chain(ticker_sf)
            S0 = float(get_close_price(ticker_sf).iloc[-1])
            r = 0.04  # si tu veux tu peux en faire un input

            expiries = sorted(chains.keys())[:n_mat]

            maturities = []
            for e in expiries:
                T = (datetime.strptime(e, "%Y-%m-%d") - datetime.today()).days / 365.0
                maturities.append(T)

            # =========================
            # 2) Strikes & price matrix
            # =========================
            all_strikes = sorted(
                set().union(*[set(chains[e]["strike"]) for e in expiries])
            )
            strikes = np.array(all_strikes, dtype=float)

            price_matrix = np.zeros((len(maturities), len(strikes))) * np.nan

            for i, e in enumerate(expiries):
                df = chains[e].copy()
                df = df.set_index("strike")

                for j, K in enumerate(strikes):
                    if K in df.index:
                        price_matrix[i, j] = float(df.loc[K]["lastPrice"])

            # On enlève les colonnes entièrement NaN
            valid_cols = ~np.isnan(price_matrix).all(axis=0)
            strikes = strikes[valid_cols]
            price_matrix = price_matrix[:, valid_cols]

            # =========================
            # 3) Calibrateur BS -> surface de vol
            # =========================
            cal = Calibrator(
                strikes=strikes,
                maturities=maturities,
                S0=S0,
                r=r,
                price_matrix=price_matrix
            )

            vol_surface = cal.build_surface()  # objet VolatilitySurface

            # surface brute
            raw_matrix = vol_surface.surface  # shape (len(maturities), len(strikes))

            df_raw = pd.DataFrame(
                raw_matrix,
                index=maturities,
                columns=strikes
            )

            st.subheader("Raw implied volatility surface")
            st.dataframe(df_raw)

            # =========================
            # 4) Interpolation (simple bilinéaire)
            # =========================
            if interp_method == "Bilinear (2D interpolation)":
                df_interp = (
                    df_raw
                    .interpolate(axis=1, limit_direction="both")
                    .interpolate(axis=0, limit_direction="both")
                )
            else:
                df_interp = df_raw.copy()

            st.subheader("Interpolated surface used for plots")
            st.dataframe(df_interp)

            # =========================
            # 5) Heatmap 2D
            # =========================
            st.subheader("Heatmap (vol vs strike & maturity)")

            fig_hm, ax_hm = plt.subplots(figsize=(8, 5))
            cax = ax_hm.imshow(
                df_interp.values,
                aspect="auto",
                origin="lower",
                extent=[strikes.min(), strikes.max(), min(maturities), max(maturities)]
            )
            ax_hm.set_xlabel("Strike")
            ax_hm.set_ylabel("Maturity (years)")
            fig_hm.colorbar(cax, label="Implied Volatility")
            st.pyplot(fig_hm)

            # =========================
            # 6) 3D surface (optionnel)
            # =========================
            if show_3d:
                st.subheader("3D volatility surface")

                K_grid, T_grid = np.meshgrid(strikes, maturities)
                Z = df_interp.values

                fig3d = plt.figure(figsize=(8, 5))
                ax3d = fig3d.add_subplot(111, projection="3d")
                surf = ax3d.plot_surface(K_grid, T_grid, Z, cmap="viridis")
                ax3d.set_xlabel("Strike")
                ax3d.set_ylabel("Maturity (years)")
                ax3d.set_zlabel("Implied Volatility")
                fig3d.colorbar(surf, shrink=0.5, aspect=10)
                st.pyplot(fig3d)

            # =========================
            # 7) Skew & term structure
            # =========================
            st.subheader("Skew & Term Structure analysis")

            colA, colB = st.columns(2)

            with colA:
                # Skew: vol(K) pour une maturité choisie
                chosen_T = st.selectbox(
                    "Maturity for skew (years)",
                    options=list(df_interp.index),
                    format_func=lambda x: f"{x:.4f}",
                    key="surf_skew_T"
                )

                vol_skew = df_interp.loc[chosen_T]

                fig_skew, ax_skew = plt.subplots(figsize=(6, 4))
                ax_skew.plot(df_interp.columns, vol_skew)
                ax_skew.set_xlabel("Strike")
                ax_skew.set_ylabel("Implied Volatility")
                ax_skew.set_title(f"Skew at T = {chosen_T:.4f} years")
                st.pyplot(fig_skew)

            with colB:
                # Term structure: vol(T) pour un strike choisi
                chosen_K = st.selectbox(
                    "Strike for term structure",
                    options=list(df_interp.columns),
                    key="surf_term_K"
                )

                vol_term = df_interp[chosen_K]

                fig_term, ax_term = plt.subplots(figsize=(6, 4))
                ax_term.plot(df_interp.index, vol_term)
                ax_term.set_xlabel("Maturity (years)")
                ax_term.set_ylabel("Implied Volatility")
                ax_term.set_title(f"Term structure at K = {chosen_K:.2f}")
                st.pyplot(fig_term)

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


from pricer.models.bs_greeks import (
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put
)

with tab_greeks:

    st.subheader("Black-Scholes Greeks")

    col1, col2 = st.columns(2)

    with col1:
        S = st.number_input("Spot S₀", value=100.0)
        K = st.number_input("Strike K", value=100.0)
        T = st.number_input("Maturity T (years)", value=1.0)
    
    with col2:
        r = st.number_input("Risk-free rate r", value=0.04)
        sigma = st.number_input("Volatility σ", value=0.20)
        opt_type = st.radio("Option type", ["Call", "Put"], horizontal=True)

    if st.button("Compute Greeks"):

        try:
            if opt_type == "Call":
                delta = delta_call(S, K, r, sigma, T)
                theta = theta_call(S, K, r, sigma, T)
                rho   = rho_call(S, K, r, sigma, T)
            else:
                delta = delta_put(S, K, r, sigma, T)
                theta = theta_put(S, K, r, sigma, T)
                rho   = rho_put(S, K, r, sigma, T)

            g = gamma(S, K, r, sigma, T)
            v = vega(S, K, r, sigma, T)

            st.success("Greeks computed successfully!")

            st.write(f"**Delta:** {delta:.6f}")
            st.write(f"**Gamma:** {g:.6f}")
            st.write(f"**Vega:** {v:.6f}")
            st.write(f"**Theta:** {theta:.6f}")
            st.write(f"**Rho:** {rho:.6f}")

        except Exception as e:
            st.error(f"Error while computing Greeks: {e}")
