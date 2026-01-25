import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricer.models.bs_greeks import (
    delta_call, delta_put,
    gamma, vega,
    theta_call, theta_put,
    rho_call, rho_put
)

def render():

    st.subheader("Greeks")

    # =====================================================
    # 1) Greeks at a given point
    # =====================================================
    st.markdown("### 1) Greeks at a given point")

    col1, col2 = st.columns(2)

    with col1:
        S = st.number_input("Spot S₀", value=100.0, min_value=1e-8, key="g_S")
        K = st.number_input("Strike K", value=100.0, min_value=1e-8, key="g_K")
        T = st.number_input("Maturity T (years)", value=1.0, min_value=1e-6, key="g_T")

    with col2:
        r = st.number_input("Risk-free rate r", value=0.04, key="g_r")
        sigma = st.number_input("Volatility σ", value=0.20, min_value=1e-8, key="g_sigma")
        opt_type = st.radio("Option type", ["Call", "Put"], key="g_type")

    if st.button("Compute Greeks", key="g_compute"):

        if opt_type == "Call":
            delta = delta_call(S, K, r, sigma, T)
            theta = theta_call(S, K, r, sigma, T)
            rho   = rho_call(S, K, r, sigma, T)
        else:
            delta = delta_put(S, K, r, sigma, T)
            theta = theta_put(S, K, r, sigma, T)
            rho   = rho_put(S, K, r, sigma, T)

        gamma_val = gamma(S, K, r, sigma, T)
        vega_val  = vega(S, K, r, sigma, T)

        c1, c2, c3 = st.columns(3)
        c1.metric("Delta", f"{delta:.6f}")
        c2.metric("Gamma", f"{gamma_val:.6f}")
        c3.metric("Vega",  f"{vega_val:.6f}")

        c4, c5 = st.columns(2)
        c4.metric("Theta", f"{theta:.6f}")
        c5.metric("Rho",   f"{rho:.6f}")

    # =====================================================
    # 2) Greeks vs Strike
    # =====================================================
    st.markdown("---")
    st.markdown("### 2) Greeks vs Strike")

    k_min = st.slider("Min K factor", 0.3, 1.0, 0.5, key="g_kmin")
    k_max = st.slider("Max K factor", 1.0, 2.0, 1.5, key="g_kmax")
    n_pts = st.slider("Grid points", 20, 300, 80, key="g_kpts")

    if st.button("Plot Greeks vs Strike", key="g_plot_k"):

        K_grid = np.linspace(k_min * K, k_max * K, n_pts)

        delta_g, gamma_g, vega_g = [], [], []

        for Ki in K_grid:
            if opt_type == "Call":
                delta_g.append(delta_call(S, Ki, r, sigma, T))
            else:
                delta_g.append(delta_put(S, Ki, r, sigma, T))

            gamma_g.append(gamma(S, Ki, r, sigma, T))
            vega_g.append(vega(S, Ki, r, sigma, T))

        fig, axs = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

        axs[0].plot(K_grid, delta_g)
        axs[0].set_title("Delta vs Strike")
        axs[0].grid(True)

        axs[1].plot(K_grid, gamma_g)
        axs[1].set_title("Gamma vs Strike")
        axs[1].grid(True)

        axs[2].plot(K_grid, vega_g)
        axs[2].set_title("Vega vs Strike")
        axs[2].set_xlabel("Strike")
        axs[2].grid(True)

        st.pyplot(fig)

    # =====================================================
    # 3) Gamma heatmap Γ(S, K)
    # =====================================================
    st.markdown("---")
    st.markdown("### 3) Gamma heatmap Γ(S, K)")

    s_min = st.slider("Min S factor", 0.5, 1.0, 0.7, key="g_smin")
    s_max = st.slider("Max S factor", 1.0, 1.5, 1.3, key="g_smax")

    if st.button("Plot Gamma Heatmap", key="g_heatmap"):

        S_vals = np.linspace(s_min * S, s_max * S, 40)
        K_vals = np.linspace(0.7 * K, 1.3 * K, 40)

        G = np.zeros((len(S_vals), len(K_vals)))

        for i, Si in enumerate(S_vals):
            for j, Kj in enumerate(K_vals):
                G[i, j] = gamma(Si, Kj, r, sigma, T)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            G,
            origin="lower",
            aspect="auto",
            extent=[K_vals[0], K_vals[-1], S_vals[0], S_vals[-1]]
        )
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Spot S")
        ax.set_title("Gamma heatmap Γ(S, K)")
        fig.colorbar(im, ax=ax)

        st.pyplot(fig)
