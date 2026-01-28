import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricer.models.bs_greeks import (
    delta_call, delta_put,
    gamma, vega,
    theta_call, theta_put,
    rho_call, rho_put
)

# --- FONCTIONS CACHÉES POUR LES CALCULS ---

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

# --- FONCTION PRINCIPALE DE RENDU ---

def render():
    st.subheader("Greeks Analysis")

    # Sidebar ou colonnes pour les inputs globaux
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Spot S₀", value=100.0, min_value=1e-8, key="g_S")
        K = st.number_input("Strike K", value=100.0, min_value=1e-8, key="g_K")
        T = st.number_input("Maturity T (years)", value=1.0, min_value=1e-6, key="g_T")
    with col2:
        r = st.number_input("Risk-free rate r", value=0.04, key="g_r")
        sigma = st.number_input("Volatility σ", value=0.20, min_value=1e-8, key="g_sigma")
        opt_type = st.radio("Option type", ["Call", "Put"], key="g_type")

    # =====================================================
    # 1) Greeks at a given point
    # =====================================================
    st.markdown("---")
    st.markdown("### 1) Greeks at a given point")

    if st.button("Compute Greeks", key="g_compute"):
        greeks = compute_single_greeks(S, K, T, r, sigma, opt_type)
        st.session_state.single_greeks = greeks

    if 'single_greeks' in st.session_state:
        res = st.session_state.single_greeks
        c1, c2, c3 = st.columns(3)
        c1.metric("Delta", f"{res['Delta']:.6f}")
        c2.metric("Gamma", f"{res['Gamma']:.6f}")
        c3.metric("Vega",  f"{res['Vega']:.6f}")
        c4, c5 = st.columns(2)
        c4.metric("Theta", f"{res['Theta']:.6f}")
        c5.metric("Rho",   f"{res['Rho']:.6f}")

    # =====================================================
    # 2) Greeks vs Strike
    # =====================================================
    st.markdown("---")
    st.markdown("### 2) Greeks vs Strike")

    col_s1, col_s2, col_s3 = st.columns(3)
    k_min = col_s1.slider("Min K factor", 0.3, 1.0, 0.5, key="g_kmin")
    k_max = col_s2.slider("Max K factor", 1.0, 2.0, 1.5, key="g_kmax")
    n_pts = col_s3.slider("Grid points", 20, 300, 80, key="g_kpts")

    if st.button("Plot Greeks vs Strike", key="g_plot_k"):
        grid_data = compute_greeks_grid(S, K, T, r, sigma, opt_type, k_min, k_max, n_pts)
        st.session_state.greeks_grid = grid_data

    if 'greeks_grid' in st.session_state:
        K_grid, delta_g, gamma_g, vega_g = st.session_state.greeks_grid
        col1, col2, col3 = st.columns(3)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.plot(K_grid, delta_g)
            ax1.set_title("Delta vs Strike")
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(K_grid, gamma_g, color='orange')
            ax2.set_title("Gamma vs Strike")
            st.pyplot(fig2)
        with col3:
            fig3, ax3 = plt.subplots()
            ax3.plot(K_grid, vega_g, color='green')
            ax3.set_title("Vega vs Strike")
            st.pyplot(fig3)

    # =====================================================
    # 3) Gamma heatmap Γ(S, K)
    # =====================================================
    st.markdown("---")
    st.markdown("### 3) Gamma heatmap Γ(S, K)")

    s_min = st.slider("Min S factor", 0.5, 1.0, 0.7, key="g_smin")
    s_max = st.slider("Max S factor", 1.0, 1.5, 1.3, key="g_smax")

    if st.button("Plot Gamma Heatmap", key="g_heatmap"):
        heatmap_data = compute_gamma_heatmap(S, K, T, r, sigma, s_min, s_max)
        st.session_state.gamma_heatmap = heatmap_data

    if 'gamma_heatmap' in st.session_state:
        S_vals, K_vals, G = st.session_state.gamma_heatmap
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            G, origin="lower", aspect="auto",
            extent=[K_vals[0], K_vals[-1], S_vals[0], S_vals[-1]]
        )
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Spot S")
        ax.set_title("Gamma heatmap Γ(S, K)")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
