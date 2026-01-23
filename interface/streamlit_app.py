
# python3 -m streamlit run interface/streamlit_app.py

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
#  HELPER
# -------------------------

def choose_next_expiry(chains):
    """Retourne (expiry_str, days, T) pour la prochaine échéance future."""
    if not chains:
        return None, None, None

    expiries = sorted(chains.keys())
    today = datetime.today().date()

    best_expiry, best_days = None, None
    for e in expiries:
        d = datetime.strptime(e, "%Y-%m-%d").date()
        days = (d - today).days
        if days <= 0:
            continue
        if best_days is None or days < best_days:
            best_days = days
            best_expiry = e

    if best_expiry is None:
        return None, None, None

    return best_expiry, best_days, best_days / 365.0


def choose_expiry_closest_to_T(chains, T_target):
    """Retourne (expiry_str, days, T) pour l'échéance la plus proche de T_target."""
    if not chains or T_target is None or T_target <= 0:
        return None, None, None

    expiries = sorted(chains.keys())
    today = datetime.today().date()

    best_expiry, best_days, best_diff = None, None, None

    for e in expiries:
        d = datetime.strptime(e, "%Y-%m-%d").date()
        days = (d - today).days
        if days <= 0:
            continue
        T = days / 365.0
        diff = abs(T - T_target)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_expiry = e
            best_days = days

    if best_expiry is None:
        return None, None, None

    return best_expiry, best_days, best_days / 365.0


def get_market_call_price_from_chain(chains, expiry, K):
    """
    Récupère un prix de CALL depuis une option chain Yahoo Finance
    en prenant le strike le plus proche de K.
    """
    if chains is None or expiry not in chains:
        return None, None

    df = chains[expiry].copy()

    if "strike" not in df.columns:
        return None, None

    # strike le plus proche
    df = df.sort_values("strike")
    idx = (df["strike"] - K).abs().idxmin()
    row = df.loc[idx]

    price = None

    # priorité : lastPrice
    if "lastPrice" in row and row["lastPrice"] is not None and row["lastPrice"] > 0:
        price = float(row["lastPrice"])

    # fallback : mid bid/ask
    elif "bid" in row and "ask" in row:
        bid, ask = row["bid"], row["ask"]
        if bid is not None and ask is not None and ask >= bid:
            price = float(0.5 * (bid + ask))

    return price, float(row["strike"])

# -------------------------
#   TABS
# -------------------------
tab_price, tab_smile, tab_surface, tab_heston, tab_greeks, tab_linear, tab_replication, tab_swaps= st.tabs(
    ["Pricer", "Volatility Smile", "Volatility Surface", "Heston Monte Carlo", "Greeks", "Linear","Replication", "Swaps"]
)

with tab_price:

    st.header("Pricing")

    # ============================
    #       TICKER
    # ============================
    ticker = st.text_input("Enter a ticker (e.g. SPY, AAPL, MSFT)", key="pr_ticker")
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
        opt_type = st.radio("Option type", ["Call", "Put"], key="pr_opt_type")

        use_atm_strike = st.checkbox("Use ATM strike (K ≈ Spot)", value=False, key="pr_atm")
        if use_atm_strike:
            K = st.number_input("Strike K", value=float(round(S0_default, 2)),
                                disabled=True, key="pr_K_atm")
        else:
            K = st.number_input("Strike K", value=0.0, key="pr_K")

        use_auto_maturity = st.checkbox("Use next option expiry from market", value=False, key="pr_auto_T")
        expiry_used = None

        if use_auto_maturity and chains is not None:
            expiry_used, days_auto, T_auto = choose_next_expiry(chains)
            if expiry_used is not None:
                st.info(f"Using next expiry {expiry_used} (~{days_auto} days).")
                T = st.number_input("Maturity T (years)", value=float(round(T_auto, 4)),
                                    disabled=True, key="pr_T_auto")
            else:
                st.warning("No future expiry found.")
                T = st.number_input("Maturity T (years)", value=0.5, key="pr_T_manual_fallback")
        else:
            T = st.number_input("Maturity T (years)", value=0.5, key="pr_T")

    # ----- MARKET SETTINGS -----
    with col2:
        use_auto_spot = st.checkbox("Use spot from ticker", value=bool(ticker), key="pr_auto_spot")
        if use_auto_spot:
            S0 = st.number_input("Spot S₀", value=float(round(S0_default, 4)),
                                 disabled=True, key="pr_S0_auto")
        else:
            S0 = st.number_input("Spot S₀", value=float(round(S0_default, 4)),
                                 key="pr_S0")

        r = st.number_input("Risk-free rate r", value=0.04, key="pr_r")

    # ============================
    #   VOLATILITY METHOD
    # ============================
    st.subheader("Volatility")

    vol_mode = st.radio(
        "Volatility method",
        ["Manual volatility", "Implied volatility (Newton)"],
        key="pr_vol_mode"
    )

    sigma = None
    market_price = None
    use_auto_market_price = False

    if vol_mode == "Manual volatility":
        sigma = st.number_input("Volatility σ", value=0.20, key="pr_sigma_manual")

    else:
        st.markdown("**Implied volatility (Newton)** — requires a market price.")
        if opt_type == "Put":
            st.warning("Implied vol solver implemented only for CALLs.")

        use_auto_market_price = st.checkbox(
            "Use market price from option chain",
            value=(chains is not None),
            key="pr_use_chain_price"
        )

        if use_auto_market_price and chains is not None:

            expiry_for_price = expiry_used or choose_expiry_closest_to_T(chains, T)[0]

            price_auto = None
            if expiry_for_price and K > 0:
                price_auto, K_used = get_market_call_price_from_chain(
                    chains, expiry_for_price, K
                )

            if price_auto is not None:
                st.info(f"Market CALL price {price_auto:.4f} (expiry {expiry_for_price}, strike {K_used})")
                market_price = st.number_input(
                    "Market option price",
                    value=float(round(price_auto, 4)),
                    disabled=True,
                    key="pr_market_price_auto"
                )
            else:
                st.warning("No suitable option found — enter price manually.")
                use_auto_market_price = False

        if not use_auto_market_price:
            market_price = st.number_input("Market price", value=1.0, key="pr_market_price")

    # ============================
    #       CALCULATE PRICE
    # ============================
    if st.button("Calculate price", key="pr_btn_calc"):

        errors = []
        if T <= 0:
            errors.append("Maturity T must be > 0.")
        if K <= 0:
            errors.append("Strike K must be > 0.")
        if S0 <= 0:
            errors.append("Spot S₀ must be > 0.")
        if vol_mode == "Implied volatility (Newton)" and (
            market_price is None or market_price <= 0):
            errors.append("Market price must be > 0.")

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        opt = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

        if vol_mode == "Implied volatility (Newton)":

            intrinsic = max(S0 - K, 0) if opt_type == "Call" else max(K - S0, 0)

            if market_price < intrinsic:
                st.error(f"Market price {market_price} < intrinsic value {intrinsic}. Impossible IV.")
                st.stop()

            if market_price > S0:
                st.error("Market price cannot exceed spot price.")
                st.stop()

            solver = NewtonImpliedVolSolver()
            opt_solver = MarketOption(S0=S0, K=K, T=T, r=r, price_mkt=market_price)
            sigma = solver.solve(opt_solver)

            if sigma is None or not np.isfinite(sigma):
                st.error("Implied volatility could not be computed.")
                st.stop()

            st.success(f"Implied volatility: {sigma:.4f}")

        market = MarketData(spot=S0, r=r)
        model = BlackScholesModel(market_data=market, sigma=sigma)
        engine = PricingEngine(model=model)

        price = engine.price_european(opt, kind=opt_type.lower())
        st.success(f"Option price: {price:.4f}")


with tab_heston:

    st.subheader("Heston Monte Carlo")

    # ===========================
    # Paramètres du modèle
    # ===========================
    col1, col2 = st.columns(2)

    with col1:
        S0 = st.number_input("Spot S₀", value=100.0, key="h_S0")
        r = st.number_input("Risk-free rate r", value=0.04, key="h_r")
        T = st.number_input("Maturity T (years)", value=1.0, key="h_T")
        K = st.number_input("Strike K", value=100.0, key="h_K")
        opt_type = st.radio("Option type", ["Call", "Put"], key="h_opt_type")

    with col2:
        v0 = st.number_input("Initial variance v₀", value=0.04, key="h_v0")
        kappa = st.number_input("Mean reversion κ", value=1.5, key="h_kappa")
        theta = st.number_input("Long-run variance θ", value=0.04, key="h_theta")
        sigma_v = st.number_input("Vol of variance σᵥ", value=0.3, key="h_sigma_v")
        rho = st.number_input("Correlation ρ", value=-0.7, key="h_rho")
        n_steps = st.number_input("Time steps", value=100, key="h_steps")
        n_paths = st.number_input("MC paths", value=5000, key="h_paths")

    if st.button("Run Heston simulation", key="h_run"):

        try:
            # ---- Market Data ----
            market = MarketData(spot=S0, r=r)

            # ---- Option ----
            option = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

            # ---- Heston model ----
            heston = HestonModel(
                v0=v0,
                kappa=kappa,
                theta=theta,
                sigma_v=sigma_v,
                rho=rho,
                n_steps=int(n_steps),
                n_paths=int(n_paths)
            )

            # ---- Pricing ----
            price_heston = heston.price_european(option, market)
            st.success(f"Heston MC price = {price_heston:.4f}")

            # ---- (Optionnel) trajectoires ----
            S_paths = heston.simulate_paths(market, option.T)

            # Exemple : afficher une trajectoire
            fig, ax = plt.subplots()
            ax.plot(S_paths[0, :])
            ax.set_title("Sample Heston path")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Spot")
            st.pyplot(fig)

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

                fig = calibrator.plot_smile()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error computing smile: {e}")


with tab_surface:
    st.subheader("Volatility Surface (2D)")

    ticker_sf = st.text_input("Ticker for surface", value="SPY", key="surf_ticker")
    n_mat = st.slider("Number of maturities to use", 2, 20, 5, key="surf_mat")

    interp_method = st.selectbox(
        "Interpolation method",
        ["Bilinear (2D interpolation)"],
        key="surf_interp_method"
    )

    @st.cache_data(show_spinner=False)
    def get_calibrated_surface(ticker_sf, n_mat, r_val):
        # --- Fetch Data ---
        chains = get_option_chain(ticker_sf)
        S0 = float(get_close_price(ticker_sf).iloc[-1])
        expiries = sorted(chains.keys())[:n_mat]

        maturities_list = []
        for e in expiries:
            T = (datetime.strptime(e, "%Y-%m-%d") - datetime.today()).days / 365.0
            maturities_list.append(T)

        maturities = np.array(maturities_list)
        all_strikes = sorted(set().union(*[set(chains[e]["strike"]) for e in expiries]))
        strikes = np.array(all_strikes, dtype=float)

        # --- Build Matrix ---
        price_matrix = np.zeros((len(maturities), len(strikes))) * np.nan
        for i, e in enumerate(expiries):
            df = chains[e].set_index("strike")
            for j, K in enumerate(strikes):
                if K in df.index:
                    price_matrix[i, j] = float(df.loc[K]["lastPrice"])

        valid_cols = ~np.isnan(price_matrix).all(axis=0)
        strikes = strikes[valid_cols]
        price_matrix = price_matrix[:, valid_cols]

        # --- Calibration ---
        cal = Calibrator(strikes=strikes, maturities=maturities, S0=S0, r=r_val, price_matrix=price_matrix)
        vol_surface = cal.build_surface()

        df_raw = pd.DataFrame(vol_surface.surface, index=maturities, columns=strikes)
        # On retourne un tuple complet
        return df_raw, strikes, maturities, S0

    # LOGIQUE DE BOUTON ET SESSION
    # Si on clique sur le bouton, on force le recalcul en vidant la session spécifique
    if st.button("Compute Volatility Surface", key="surf_btn"):
        with st.status("Fetching data and computing surface...", expanded=True) as status:
            data = get_calibrated_surface(ticker_sf, n_mat, 0.04)
            st.session_state.surface_data = data
            status.update(label="Calibration Complete!", state="complete", expanded=False)

    # AFFICHAGE (Uniquement si les données existent en session)
    if 'surface_data' in st.session_state:
        try:
            # ETAPE CRUCIALE : On extrait les variables de la session à chaque exécution
            df_raw, strikes, maturities, S0 = st.session_state.surface_data

            # --- Interpolation ---
            if interp_method == "Bilinear (2D interpolation)":
                df_interp = df_raw.interpolate(axis=1, limit_direction="both").interpolate(axis=0, limit_direction="both")
            else:
                df_interp = df_raw.copy()

            st.subheader("Results")
            st.write(f"Spot Price S0: {S0:.2f}")

            # --- 3D Surface ---
            K_grid, T_grid = np.meshgrid(df_interp.columns, df_interp.index)
            fig = go.Figure(data=[go.Surface(z=df_interp.values, x=K_grid, y=T_grid, colorscale='Viridis')])
            fig.update_layout(
                scene=dict(xaxis_title='Strike', yaxis_title='Maturity', zaxis_title='Vol'),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            colA, colB = st.columns(2)

            with colA:
                chosen_T = st.selectbox("Maturity for skew", options=list(df_interp.index), format_func=lambda x: f"{x:.4f}")
                fig_skew, ax_skew = plt.subplots()
                ax_skew.plot(df_interp.columns, df_interp.loc[chosen_T], marker='o')
                ax_skew.set_title(f"Skew (T={chosen_T:.4f})")
                st.pyplot(fig_skew)

            with colB:
                chosen_K = st.selectbox("Strike for term structure", options=list(df_interp.columns))
                fig_term, ax_term = plt.subplots()
                ax_term.plot(df_interp.index, df_interp[chosen_K], marker='o', color='orange')
                ax_term.set_title(f"Term Structure (K={chosen_K:.2f})")
                st.pyplot(fig_term)

        except Exception as e:
            st.error(f"Erreur d'affichage : {e}")

from pricer.models.bs_greeks import (
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put
)

with tab_greeks:

    st.subheader("Black–Scholes Greeks")

    # ===========================
    # 1) Paramètres de base
    # ===========================
    col1, col2 = st.columns(2)

    with col1:
        S = st.number_input("Spot S₀", value=100.0, key="g_S0")
        K = st.number_input("Strike K", value=100.0, key="g_K0")
        T = st.number_input("Maturity T (years)", value=1.0, key="g_T0")

    with col2:
        r = st.number_input("Risk-free rate r", value=0.04, key="g_r0")
        sigma = st.number_input("Volatility σ", value=0.20, key="g_sigma0")
        opt_type = st.radio("Option type", ["Call", "Put"], key="g_opt_type0")

    if st.button("Compute BS Greeks", key="g_btn_bs"):

        try:
            if opt_type == "Call":
                delta_val = delta_call(S, K, r, sigma, T)
                theta_val = theta_call(S, K, r, sigma, T)
                rho_val   = rho_call(S, K, r, sigma, T)
            else:
                delta_val = delta_put(S, K, r, sigma, T)
                theta_val = theta_put(S, K, r, sigma, T)
                rho_val   = rho_put(S, K, r, sigma, T)

            gamma_val = gamma(S, K, r, sigma, T)
            vega_val  = vega(S, K, r, sigma, T)

            st.success("Greeks computed at (S, K, T)")

            st.write(f"**Delta:** {delta_val:.6f}")
            st.write(f"**Gamma:** {gamma_val:.6f}")
            st.write(f"**Vega:**  {vega_val:.6f}")
            st.write(f"**Theta:** {theta_val:.6f}")
            st.write(f"**Rho:**   {rho_val:.6f}")

        except Exception as e:
            st.error(f"Error computing BS Greeks: {e}")


    st.markdown("---")
    st.subheader("BS Greeks profiles vs Strike")

    # ===========================
    # 2) Profils des Greeks vs K
    # ===========================
    colK1, colK2 = st.columns(2)

    with colK1:
        k_min_factor = st.number_input("Min strike factor (×K)", value=0.5, key="g_k_min_factor")
        k_max_factor = st.number_input("Max strike factor (×K)", value=1.5, key="g_k_max_factor")

    with colK2:
        n_points = st.number_input("Number of strikes", value=50, min_value=10, max_value=300, key="g_k_npoints")

    if st.button("Plot BS Greeks vs Strike", key="g_btn_greeks_vs_k"):

        try:
            if K <= 0 or S <= 0 or T <= 0 or sigma <= 0:
                st.error("S, K, T, σ must be strictly positive.")
                st.stop()

            K_min = max(1e-6, k_min_factor * K)
            K_max = max(K_min + 1e-6, k_max_factor * K)

            K_grid = np.linspace(K_min, K_max, int(n_points))

            delta_list = []
            gamma_list = []
            vega_list  = []
            theta_list = []
            rho_list   = []

            for K_i in K_grid:
                if opt_type == "Call":
                    delta_i = delta_call(S, K_i, r, sigma, T)
                    theta_i = theta_call(S, K_i, r, sigma, T)
                    rho_i   = rho_call(S, K_i, r, sigma, T)
                else:
                    delta_i = delta_put(S, K_i, r, sigma, T)
                    theta_i = theta_put(S, K_i, r, sigma, T)
                    rho_i   = rho_put(S, K_i, r, sigma, T)

                gamma_i = gamma(S, K_i, r, sigma, T)
                vega_i  = vega(S, K_i, r, sigma, T)

                delta_list.append(delta_i)
                gamma_list.append(gamma_i)
                vega_list.append(vega_i)
                theta_list.append(theta_i)
                rho_list.append(rho_i)

            # Plot Delta
            fig_d, ax_d = plt.subplots(figsize=(6,4))
            ax_d.plot(K_grid, delta_list)
            ax_d.set_title("Delta vs Strike")
            ax_d.set_xlabel("Strike K")
            ax_d.set_ylabel("Delta")
            st.pyplot(fig_d)

            # Plot Gamma
            fig_g, ax_g = plt.subplots(figsize=(6,4))
            ax_g.plot(K_grid, gamma_list)
            ax_g.set_title("Gamma vs Strike")
            ax_g.set_xlabel("Strike K")
            ax_g.set_ylabel("Gamma")
            st.pyplot(fig_g)

            # Plot Vega
            fig_v, ax_v = plt.subplots(figsize=(6,4))
            ax_v.plot(K_grid, vega_list)
            ax_v.set_title("Vega vs Strike")
            ax_v.set_xlabel("Strike K")
            ax_v.set_ylabel("Vega")
            st.pyplot(fig_v)

            # Plot Theta
            fig_t, ax_t = plt.subplots(figsize=(6,4))
            ax_t.plot(K_grid, theta_list)
            ax_t.set_title("Theta vs Strike")
            ax_t.set_xlabel("Strike K")
            ax_t.set_ylabel("Theta")
            st.pyplot(fig_t)

            # Plot Rho
            fig_r, ax_r = plt.subplots(figsize=(6,4))
            ax_r.plot(K_grid, rho_list)
            ax_r.set_title("Rho vs Strike")
            ax_r.set_xlabel("Strike K")
            ax_r.set_ylabel("Rho")
            st.pyplot(fig_r)

            # Export CSV
            df_greeks = pd.DataFrame({
                "K": K_grid,
                "Delta": delta_list,
                "Gamma": gamma_list,
                "Vega": vega_list,
                "Theta": theta_list,
                "Rho": rho_list
            })

            csv = df_greeks.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Greeks vs Strike as CSV",
                data=csv,
                file_name="bs_greeks_vs_strike.csv",
                mime="text/csv",
                key="g_dl_greeks_csv"
            )

        except Exception as e:
            st.error(f"Error computing Greeks profiles: {e}")


    st.markdown("---")
    st.subheader("Gamma heatmap Γ(S, K)")

    # ===========================
    # 3) Heatmap Gamma(S,K)
    # ===========================
    colH1, colH2 = st.columns(2)

    with colH1:
        s_min_factor = st.number_input("Min spot factor (×S₀)", value=0.5, key="g_s_min_factor")
        s_max_factor = st.number_input("Max spot factor (×S₀)", value=1.5, key="g_s_max_factor")

    with colH2:
        grid_S = st.number_input("Grid size S dimension", value=30, min_value=10, max_value=100, key="g_s_grid")
        grid_K = st.number_input("Grid size K dimension", value=30, min_value=10, max_value=100, key="g_k_grid_heat")

    if st.button("Plot Gamma heatmap", key="g_btn_gamma_heat"):

        try:
            S_min = max(1e-6, s_min_factor * S)
            S_max = max(S_min + 1e-6, s_max_factor * S)
            K_min = max(1e-6, 0.5 * K)
            K_max = max(K_min + 1e-6, 1.5 * K)

            S_vals = np.linspace(S_min, S_max, int(grid_S))
            K_vals = np.linspace(K_min, K_max, int(grid_K))

            gamma_matrix = np.zeros((len(S_vals), len(K_vals)))

            for i, S_i in enumerate(S_vals):
                for j, K_j in enumerate(K_vals):
                    gamma_matrix[i, j] = gamma(S_i, K_j, r, sigma, T)

            fig_h, ax_h = plt.subplots(figsize=(7,5))
            cax = ax_h.imshow(
                gamma_matrix,
                origin="lower",
                aspect="auto",
                extent=[K_vals[0], K_vals[-1], S_vals[0], S_vals[-1]]
            )
            ax_h.set_xlabel("Strike K")
            ax_h.set_ylabel("Spot S")
            ax_h.set_title("Gamma heatmap Γ(S, K)")
            fig_h.colorbar(cax, label="Gamma")
            st.pyplot(fig_h)

        except Exception as e:
            st.error(f"Error plotting Gamma heatmap: {e}")


    st.markdown("---")
    st.subheader("Heston Greeks (Monte Carlo bump)")

    # ===========================
    # 4) Heston MC Greeks
    # ===========================
    colHes1, colHes2 = st.columns(2)

    with colHes1:
        S0_h = st.number_input("Heston S₀", value=100.0, key="hg_S0")
        K_h = st.number_input("Heston Strike K", value=100.0, key="hg_K")
        T_h = st.number_input("Heston T (years)", value=1.0, key="hg_T")
        r_h = st.number_input("Heston r", value=0.04, key="hg_r")
        opt_type_h = st.radio("Heston option type", ["Call", "Put"], key="hg_opt_type")

    with colHes2:
        v0_h = st.number_input("Heston v₀", value=0.04, key="hg_v0")
        kappa_h = st.number_input("κ", value=1.5, key="hg_kappa")
        theta_h = st.number_input("θ", value=0.04, key="hg_theta")
        sigma_v_h = st.number_input("σᵥ", value=0.3, key="hg_sigma_v")
        rho_h = st.number_input("ρ", value=-0.7, key="hg_rho")
        n_steps_h = st.number_input("Time steps (Heston)", value=100, key="hg_steps")
        n_paths_h = st.number_input("MC paths (Heston)", value=5000, key="hg_paths")

    colBump1, colBump2 = st.columns(2)
    with colBump1:
        bump_S = st.number_input("Bump for S (ΔS)", value=1.0, key="hg_bump_S")
    with colBump2:
        bump_v = st.number_input("Bump for v₀ (Δv₀)", value=0.01, key="hg_bump_v")

    def heston_price_once(S0_local, v0_local):
        market_h = MarketData(spot=S0_local, r=r_h)
        market_h.T = float(T_h)
        market_h.v_0 = float(v0_local)
        market_h.kappa = float(kappa_h)
        market_h.theta = float(theta_h)
        market_h.sigma_v = float(sigma_v_h)
        market_h.rho = float(rho_h)
        market_h.n_steps = int(n_steps_h)
        market_h.n_paths = int(n_paths_h)

        if opt_type_h == "Call":
            opt_h = EuropeanCall(K_h, T_h)
        else:
            opt_h = EuropeanPut(K_h, T_h)

        model_h = HestonModel()
        price_h = model_h.price_european(opt_h, market_h)
        return price_h

    if st.button("Compute Heston MC Greeks", key="hg_btn"):

        try:
            # Prix de base
            P0 = heston_price_once(S0_h, v0_h)

            # Delta & Gamma via bump sur S
            P_up_S = heston_price_once(S0_h + bump_S, v0_h)
            P_dn_S = heston_price_once(S0_h - bump_S, v0_h)

            delta_h = (P_up_S - P_dn_S) / (2 * bump_S)
            gamma_h = (P_up_S - 2*P0 + P_dn_S) / (bump_S**2)

            # "Vega" Heston via bump sur v0
            P_up_v = heston_price_once(S0_h, v0_h + bump_v)
            P_dn_v = heston_price_once(S0_h, v0_h - bump_v)
            vega_h = (P_up_v - P_dn_v) / (2 * bump_v)

            st.success("Heston MC Greeks computed:")

            st.write(f"**Delta (MC, Heston):** {delta_h:.6f}")
            st.write(f"**Gamma (MC, Heston):** {gamma_h:.6f}")
            st.write(f"**Vega wrt v₀ (MC, Heston):** {vega_h:.6f}")

        except Exception as e:
            st.error(f"Error computing Heston MC Greeks: {e}")


with tab_linear:

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

with tab_replication:

    st.header("Replication & Hedging playground (Black–Scholes)")

    # ----------------------------
    # Inputs
    # ----------------------------
    col1, col2 = st.columns(2)

    with col1:
        S0 = st.number_input("Spot S₀", value=100.0, key="rep_S0")
        K  = st.number_input("Strike K", value=100.0, key="rep_K")
        T  = st.number_input("Maturity T (years)", value=1.0, min_value=1e-4, key="rep_T")
        opt_type = st.radio("Option", ["Call", "Put"], key="rep_type")

    with col2:
        r = st.number_input("Risk-free rate r", value=0.04, key="rep_r")
        sigma = st.number_input("Volatility σ", value=0.20, min_value=1e-4, key="rep_sigma")
        q = st.number_input("Dividend yield q (optional)", value=0.0, key="rep_q")  # si tu ne gères pas q, laisse 0

    st.markdown("---")

    # ----------------------------
    # Choose "hedge spot" (where delta is computed)
    # ----------------------------
    st.subheader("Static replication (one delta computed at a chosen spot)")

    hedge_spot_mode = st.radio(
        "Delta computed at:",
        ["Current spot S₀", "Custom spot"],
        key="rep_hedge_spot_mode"
    )

    if hedge_spot_mode == "Current spot S₀":
        S_hedge = float(S0)
        st.info(f"Delta will be computed at S = {S_hedge:.4f}")
    else:
        S_hedge = st.number_input("Custom hedge spot", value=float(S0), min_value=1e-8, key="rep_Shedge")

    # ----------------------------
    # Compute option price & delta (use your engine for consistency)
    # ----------------------------
    # Build market & option
    market = MarketData(spot=S0, r=r, q=q) if "q" in MarketData.__init__.__code__.co_varnames else MarketData(spot=S0, r=r)

    option = EuropeanCall(K, T) if opt_type == "Call" else EuropeanPut(K, T)

    model = BlackScholesModel(market_data=market, sigma=sigma)
    engine = PricingEngine(model=model)

    # Option price today
    price0 = engine.price_european(option, kind=opt_type.lower())

    # Delta at hedge spot (use your greeks functions if you have them)
    # NOTE: I assume your greek signature is delta_call(S, K, r, sigma, T)
    if opt_type == "Call":
        delta0 = float(delta_call(S_hedge, K, r, sigma, T))
    else:
        delta0 = float(delta_put(S_hedge, K, r, sigma, T))

    # Build the static replicating portfolio:
    # Hold Δ shares and a cash position B so that:
    #   V0 = Δ*S_hedge + B = option_price0
    # => B = option_price0 - Δ*S_hedge
    B0 = price0 - delta0 * S_hedge

    # Terminal value of the portfolio at maturity (cash grows at r):
    #   V_T = Δ*S_T + B0*exp(rT)
    # (ignoring q in the cash account; for a student project, this is fine)
    # If you want to be “extra correct” with dividends, we can refine, but keep it simple.

    # ----------------------------
    # Display key numbers
    # ----------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Option price (t=0)", f"{price0:.6f}")
    c2.metric("Delta used (static)", f"{delta0:.6f}")
    c3.metric("Cash position B₀", f"{B0:.6f}")

    st.caption(
        "Static replication means you hedge once with Δ at the chosen spot. "
        "It will *not* match the payoff globally because the option is convex (Gamma)."
    )

    # ----------------------------
    # Payoff plot at maturity
    # ----------------------------
    st.subheader("Payoff comparison at maturity")

    s_min = st.number_input("Min S_T (plot)", value=0.5 * S0, min_value=1e-8, key="rep_smin")
    s_max = st.number_input("Max S_T (plot)", value=1.5 * S0, min_value=1e-8, key="rep_smax")
    n_pts = st.slider("Grid points", min_value=100, max_value=2000, value=400, key="rep_npts")

    S_grid = np.linspace(float(s_min), float(s_max), int(n_pts))

    if opt_type == "Call":
        payoff_opt = np.maximum(S_grid - K, 0.0)
    else:
        payoff_opt = np.maximum(K - S_grid, 0.0)

    payoff_rep = delta0 * S_grid + B0 * np.exp(r * T)
    err = payoff_rep - payoff_opt

    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(S_grid, payoff_opt, label="Option payoff")
    ax1.plot(S_grid, payoff_rep, label="Static replication: Δ·S_T + B·e^{rT}")
    ax1.axvline(S_hedge, linestyle="--")
    ax1.set_xlabel("Underlying at maturity S_T")
    ax1.set_ylabel("Payoff / Terminal value")
    ax1.set_title("Option vs Static Replication (one-shot delta)")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(S_grid, err)
    ax2.axhline(0.0)
    ax2.axvline(S_hedge, linestyle="--")
    ax2.set_xlabel("S_T")
    ax2.set_ylabel("Replication error (Rep - Option)")
    ax2.set_title("Replication error (convexity / gamma effect)")
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("---")

    # ==========================================================
    # Discrete delta-hedging simulation (more “quant”, still simple)
    # ==========================================================
    st.subheader("Discrete delta hedging simulation (BS)")

    with st.expander("Run a discrete hedging simulation (recommended)", expanded=True):

        colh1, colh2 = st.columns(2)
        with colh1:
            n_steps = st.number_input("Re-hedge steps (N)", value=50, min_value=1, max_value=2000, key="rep_N")
            n_paths = st.number_input("Simulation paths", value=2000, min_value=200, max_value=50000, key="rep_paths")
        with colh2:
            seed = st.number_input("Random seed", value=123, min_value=0, max_value=10_000_000, key="rep_seed")
            use_same_delta_spot = st.checkbox("Initial hedge at S₀ (ignore custom hedge spot here)", value=True, key="rep_useS0")

        if st.button("Run hedging simulation", key="rep_run_hedge"):

            try:
                np.random.seed(int(seed))

                N = int(n_steps)
                M = int(n_paths)
                dt = T / N

                # Simulate GBM under risk-neutral dynamics (ignoring q for simplicity)
                Z = np.random.normal(size=(M, N))
                S = np.zeros((M, N + 1))
                S[:, 0] = S0

                for t in range(N):
                    S[:, t+1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

                # Initial option price at t=0 (true model price)
                price_init = float(price0)

                # Initialize hedge portfolio:
                # choose initial spot for delta
                S_init_delta = float(S0) if use_same_delta_spot else float(S_hedge)

                if opt_type == "Call":
                    delta_init = np.array(delta_call(S_init_delta, K, r, sigma, T), ndmin=1).astype(float)[0]
                else:
                    delta_init = np.array(delta_put(S_init_delta, K, r, sigma, T), ndmin=1).astype(float)[0]

                # Portfolio holds delta shares + cash
                cash = price_init - delta_init * S0  # replicate at t=0
                delta_pos = np.full(M, delta_init, dtype=float)

                # Re-hedge through time
                for t in range(N):
                    # cash accrues
                    cash *= np.exp(r * dt)

                    # compute new delta at current time (time to maturity decreases)
                    tau = max(T - (t+1) * dt, 1e-12)
                    S_t = S[:, t+1]

                    if opt_type == "Call":
                        new_delta = np.array(delta_call(S_t, K, r, sigma, tau), dtype=float)
                    else:
                        new_delta = np.array(delta_put(S_t, K, r, sigma, tau), dtype=float)

                    # adjust shares: buy/sell (delta changes)
                    d_delta = new_delta - delta_pos
                    cash -= d_delta * S_t
                    delta_pos = new_delta

                # Terminal portfolio value
                V_T = delta_pos * S[:, -1] + cash

                # Terminal option payoff
                if opt_type == "Call":
                    payoff_T = np.maximum(S[:, -1] - K, 0.0)
                else:
                    payoff_T = np.maximum(K - S[:, -1], 0.0)

                hedge_error = V_T - payoff_T

                st.success("Simulation done")

                cA, cB, cC = st.columns(3)
                cA.metric("Mean hedging error", f"{hedge_error.mean():.6f}")
                cB.metric("Std hedging error", f"{hedge_error.std(ddof=1):.6f}")
                cC.metric("95% quantile |error|", f"{np.quantile(np.abs(hedge_error), 0.95):.6f}")

                # Histogram
                fig3, ax3 = plt.subplots(figsize=(7, 3.8))
                ax3.hist(hedge_error, bins=50)
                ax3.set_title("Discrete hedging error distribution (V_T - payoff)")
                ax3.set_xlabel("Hedging error")
                ax3.set_ylabel("Frequency")
                ax3.grid(True)
                st.pyplot(fig3)

                # Optional download
                df_out = pd.DataFrame({"hedging_error": hedge_error})
                st.download_button(
                    "Download hedging errors (CSV)",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="hedging_errors.csv",
                    mime="text/csv",
                    key="rep_dl_err"
                )

            except Exception as e:
                st.error(f"Hedging simulation error: {e}")
with tab_swaps:

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
