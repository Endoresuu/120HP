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