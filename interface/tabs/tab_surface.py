import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from pricer.calibration.surface_calibrator import Calibrator
from pricer.market.import_data import (
    get_option_chain,
    get_close_price
)


def render():
    st.subheader("Volatility Surface (2D)")
    st.caption("Market implied volatility surface calibrated across strikes and maturities")

    st.divider()

    # =====================================================
    # INPUTS
    # =====================================================
    col_inputs, col_outputs = st.columns([1.3, 1.7], gap="large")

    with col_inputs:
        st.subheader("Market inputs")

        ticker_sf = st.text_input(
            "Ticker for surface",
            value="SPY",
            key="surf_ticker"
        )

        n_mat = st.slider(
            "Number of maturities to use",
            2, 20, 5,
            key="surf_mat"
        )

        interp_method = st.selectbox(
            "Interpolation method",
            ["Bilinear (2D interpolation)"],
            key="surf_interp_method"
        )

        run = st.button(
            "Compute Volatility Surface",
            use_container_width=True,
            key="surf_btn"
        )

    # =====================================================
    # DATA & CALIBRATION (CACHED)
    # =====================================================
    @st.cache_data(show_spinner=False)
    def get_calibrated_surface(ticker_sf, n_mat, r_val):
        chains = get_option_chain(ticker_sf)
        S0 = float(get_close_price(ticker_sf).iloc[-1])

        today = datetime.today()

        expiries = []
        maturities = []

        for e in sorted(chains.keys()):
            T = (datetime.strptime(e, "%Y-%m-%d") - today).days / 365.0
            if T > 0:
                expiries.append(e)
                maturities.append(T)
            if len(expiries) == n_mat:
                break

        maturities = np.array(maturities)

        # strikes communs
        strikes = sorted(set().union(*[set(chains[e]["strike"]) for e in expiries]))
        strikes = np.array(strikes, dtype=float)

        # matrice de PRIX
        price_matrix = np.full((len(maturities), len(strikes)), np.nan)

        for i, e in enumerate(expiries):
            df = chains[e].set_index("strike")
            for j, K in enumerate(strikes):
                if K in df.index:
                    price_matrix[i, j] = float(df.loc[K]["lastPrice"])

        # on enlève les strikes complètement vides
        valid_cols = ~np.isnan(price_matrix).all(axis=0)
        strikes = strikes[valid_cols]
        price_matrix = price_matrix[:, valid_cols]

        # calibration
        cal = Calibrator(
            strikes=strikes,
            maturities=maturities,
            S0=S0,
            r=r_val,
            price_matrix=price_matrix
        )

        vol_surface = cal.build_surface()

        df_raw = pd.DataFrame(
            vol_surface.surface,
            index=maturities,
            columns=strikes
        )

        return df_raw, strikes, maturities, S0

    # =====================================================
    # OUTPUTS
    # =====================================================
    with col_outputs:
        st.subheader("Results")

        if run:
            with st.status(
                "Fetching data and computing surface...",
                expanded=True
            ) as status:
                try:
                    data = get_calibrated_surface(ticker_sf, n_mat, 0.04)
                    st.session_state.surface_data = data
                    status.update(
                        label="Calibration complete",
                        state="complete",
                        expanded=False
                    )
                except Exception as e:
                    status.update(label="Error", state="error", expanded=False)
                    st.error(f"Calibration error: {e}")

        # =====================================================
        # DISPLAY 
        # =====================================================
        if "surface_data" in st.session_state:
            try:
                df_raw, strikes, maturities, S0 = st.session_state.surface_data

                # --- Interpolation ---
                if interp_method == "Bilinear (2D interpolation)":
                    df_interp = (
                        df_raw
                        .dropna(how="all", axis=0)
                        .dropna(how="all", axis=1)
                        .interpolate(axis=1, limit_direction="both")
                        .interpolate(axis=0, limit_direction="both")
                    )
                else:
                    df_interp = df_raw.copy()

                st.markdown(f"**Spot price S₀**: {S0:.2f}")

                st.caption(
                    "Surface built from market option prices via implied volatility inversion "
                    "and interpolated across strikes and maturities."
                )

                # =================================================
                # 3D SURFACE
                # =================================================
                K_grid, T_grid = np.meshgrid(df_interp.columns, df_interp.index)

                fig = go.Figure(
                    data=[
                        go.Surface(
                            z=df_interp.values,
                            x=K_grid,
                            y=T_grid,
                            colorscale="Viridis"
                        )
                    ]
                )
                fig.update_layout(
                    scene=dict(
                        xaxis_title="Strike",
                        yaxis_title="Maturity",
                        zaxis_title="Volatility"
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # =================================================
                # SKEW & TERM STRUCTURE
                # =================================================
                colA, colB = st.columns(2)

                with colA:
                    chosen_T = st.selectbox(
                        "Maturity for skew",
                        options=list(df_interp.index),
                        format_func=lambda x: f"{x:.4f}"
                    )
                    fig_skew, ax_skew = plt.subplots()
                    ax_skew.plot(
                        df_interp.columns,
                        df_interp.loc[chosen_T],
                        marker="o"
                    )
                    ax_skew.set_title(f"Skew (T={chosen_T:.4f})")
                    ax_skew.set_xlabel("Strike")
                    ax_skew.set_ylabel("Volatility")
                    ax_skew.grid(True)
                    st.pyplot(fig_skew)

                with colB:
                    chosen_K = st.selectbox(
                        "Strike for term structure",
                        options=list(df_interp.columns)
                    )
                    fig_term, ax_term = plt.subplots()
                    ax_term.plot(
                        df_interp.index,
                        df_interp[chosen_K],
                        marker="o",
                        color="orange"
                    )
                    ax_term.set_title(f"Term Structure (K={chosen_K:.2f})")
                    ax_term.set_xlabel("Maturity")
                    ax_term.set_ylabel("Volatility")
                    ax_term.grid(True)
                    st.pyplot(fig_term)

            except Exception as e:
                st.error(f"Erreur d'affichage : {e}")
