import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricer.models.bs_greeks import (
    delta_call, delta_put,
    gamma, vega,
    theta_call, theta_put,
    rho_call, rho_put
)

# =====================================================
# --- FONCTIONS CACHÉES POUR LES CALCULS ---
# =====================================================

@st.cache_data
def compute_single_greeks(S, K, T, r, sigma, opt_type):
    """Calcule les Grecs pour un point unique."""
    if opt_type == "Call":
        d = delta_call(S, K, r, sigma, T)
        t = theta_call(S, K, r, sigma, T)
        rh = rho_call(S, K, r, sigma, T)
    else:
        d = delta_put(S, K, r, sigma, T)
        t = theta_put(S, K, r, sigma, T)
        rh = rho_put(S, K, r, sigma, T)

    g = gamma(S, K, r, sigma, T)
    v = vega(S, K, r, sigma, T)
    return {"Delta": d, "Gamma": g, "Vega": v, "Theta": t, "Rho": rh}


@st.cache_data
def compute_greeks_grid(S, K, T, r, sigma, opt_type, k_min, k_max, n_pts):
    """Calcule les vecteurs de Grecs pour le graphique vs Strike."""
    K_grid = np.linspace(k_min * K, k_max * K, n_pts)
    delta_g, gamma_g, vega_g = [], [], []

    for Ki in K_grid:
        if opt_type == "Call":
            delta_g.append(delta_call(S, Ki, r, sigma, T))
        else:
            delta_g.append(delta_put(S, Ki, r, sigma, T))
        gamma_g.append(gamma(S, Ki, r, sigma, T))
        vega_g.append(vega(S, Ki, r, sigma, T))

    return K_grid, delta_g, gamma_g, vega_g


@st.cache_data
def compute_gamma_heatmap(S, K, T, r, sigma, s_min, s_max):
    """Calcule la matrice pour la heatmap de Gamma."""
    S_vals = np.linspace(s_min * S, s_max * S, 40)
    K_vals = np.linspace(0.7 * K, 1.3 * K, 40)
    G = np.zeros((len(S_vals), len(K_vals)))

    for i, Si in enumerate(S_vals):
        for j, Kj in enumerate(K_vals):
            G[i, j] = gamma(Si, Kj, r, sigma, T)

    return S_vals, K_vals, G

# =====================================================
# --- FONCTION PRINCIPALE DE RENDU ---
# =====================================================

def render():
    st.subheader("Greeks Analysis")
    st.caption("Sensitivity analysis under Black–Scholes")

    st.divider()

    # =====================================================
    # GLOBAL INPUTS
    # =====================================================
    col_inputs, col_dummy = st.columns([1.3, 1])

    with col_inputs:
        st.subheader("Contract & Market")

        c1, c2 = st.columns(2)
        with c1:
            S = st.number_input("Spot S₀", value=100.0, min_value=1e-8, key="g_S")
            K = st.number_input("Strike K", value=100.0, min_value=1e-8, key="g_K")
            T = st.number_input("Maturity T (years)", value=1.0, min_value=1e-6, key="g_T")

        with c2:
            r = st.number_input("Risk-free rate r", value=0.04, key="g_r")
            sigma = st.number_input("Volatility σ", value=0.20, min_value=1e-8, key="g_sigma")
            opt_type = st.radio("Option type", ["Call", "Put"], key="g_type", horizontal=True)

    st.divider()

    # =====================================================
    # 1) Greeks at a given point
    # =====================================================
    st.markdown("### 1) Greeks at a given point")

    if st.button("Compute Greeks", key="g_compute"):
        st.session_state.single_greeks = compute_single_greeks(
            S, K, T, r, sigma, opt_type
        )

    if "single_greeks" in st.session_state:
        res = st.session_state.single_greeks

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Delta", f"{res['Delta']:.6f}")
        m2.metric("Gamma", f"{res['Gamma']:.6f}")
        m3.metric("Vega",  f"{res['Vega']:.6f}")
        m4.metric("Theta", f"{res['Theta']:.6f}")
        m5.metric("Rho",   f"{res['Rho']:.6f}")

    st.divider()

    # =====================================================
    # 2) Greeks vs Strike
    # =====================================================
    st.markdown("### 2) Greeks vs Strike")

    s1, s2, s3 = st.columns(3)
    k_min = s1.slider("Min K factor", 0.3, 1.0, 0.5, key="g_kmin")
    k_max = s2.slider("Max K factor", 1.0, 2.0, 1.5, key="g_kmax")
    n_pts = s3.slider("Grid points", 20, 300, 80, key="g_kpts")

    if st.button("Plot Greeks vs Strike", key="g_plot_k"):
        st.session_state.greeks_grid = compute_greeks_grid(
            S, K, T, r, sigma, opt_type, k_min, k_max, n_pts
        )

    if "greeks_grid" in st.session_state:
        K_grid, delta_g, gamma_g, vega_g = st.session_state.greeks_grid

        c1, c2, c3 = st.columns(3)

        with c1:
            fig, ax = plt.subplots()
            ax.plot(K_grid, delta_g)
            ax.set_title("Delta vs Strike")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots()
            ax.plot(K_grid, gamma_g)
            ax.set_title("Gamma vs Strike")
            st.pyplot(fig)

        with c3:
            fig, ax = plt.subplots()
            ax.plot(K_grid, vega_g)
            ax.set_title("Vega vs Strike")
            st.pyplot(fig)

    st.divider()

    # =====================================================
    # 3) Gamma heatmap Γ(S, K)
    # =====================================================
    st.markdown("### 3) Gamma heatmap Γ(S, K)")

    # --- Layout: left = controls, right = plot
    col_controls, col_plot = st.columns([1, 2], gap="large")

    with col_controls:
        s_min = st.slider(
            "Min S factor",
            0.5, 1.0, 0.7,
            key="g_smin"
        )

        s_max = st.slider(
            "Max S factor",
            1.0, 1.5, 1.3,
            key="g_smax"
        )

        plot_gamma = st.button("Plot Gamma Heatmap")

    if plot_gamma:
        st.session_state.gamma_heatmap = compute_gamma_heatmap(
            S, K, T, r, sigma, s_min, s_max
        )

    with col_plot:
        if "gamma_heatmap" in st.session_state:
            S_vals, K_vals, G = st.session_state.gamma_heatmap

            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(
                G,
                origin="lower",
                aspect="auto",
                extent=[K_vals[0], K_vals[-1], S_vals[0], S_vals[-1]]
            )

            ax.set_xlabel("Strike K")
            ax.set_ylabel("Spot S")
            ax.set_title("Gamma heatmap Γ(S, K)")

            fig.colorbar(im, ax=ax, shrink=0.7)
            st.pyplot(fig, use_container_width=False)
