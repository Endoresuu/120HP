import streamlit as st
from pricer.products.forward import Forward
from pricer.products.future import Future

# --- FONCTION DE CALCUL CACHÉE ---
@st.cache_data
def compute_linear_product(product_type, S0, r, q, T, K):
    """
    Effectue les calculs pour Forward ou Future.
    """
    res = {}

    if product_type == "Forward":
        fwd = Forward(K=K, T=T)
        F0 = fwd.forward_price(S0, r, q)
        V0 = fwd.value(S0, r, q)

        res = {
            "type": "Forward",
            "F0": F0,
            "V0": V0,
            "S0": S0,
            "r": r,
            "q": q,
            "T": T,
            "K": K
        }

    else: # Future
        fut = Future(T=T)
        F0 = fut.future_price(S0, r, q)

        res = {
            "type": "Future",
            "F0": F0,
            "S0": S0,
            "r": r,
            "q": q,
            "T": T
        }

    return res

# --- FONCTION DE RENDU ---
def render():
    st.header("Forwards & Futures")
    st.caption("Pricing of linear products under no-arbitrage assumptions")

    col1, col2 = st.columns(2)

    with col1:
        S0 = st.number_input("Spot S₀", value=100.0, min_value=1e-8, key="lin_S0")
        r = st.number_input("Risk-free rate r", value=0.04, key="lin_r")
        q = st.number_input("Dividend yield q", value=0.0, key="lin_q")
        T = st.number_input("Maturity T (years)", value=1.0, min_value=1e-6, key="lin_T")

    with col2:
        product_type = st.radio("Product type", ["Forward", "Future"], key="lin_type")

        K = st.number_input(
            "Delivery price K (Forward only)",
            value=100.0,
            min_value=1e-8,
            disabled=(product_type == "Future"),
            key="lin_K"
        )

    st.markdown("---")

    # BOUTON DE CALCUL
    if st.button("Price linear product", key="lin_btn"):

        # -------- Validation (reste dans l'UI) --------
        if S0 <= 0 or T <= 0:
            st.error("Spot S₀ and maturity T must be strictly positive.")
        elif product_type == "Forward" and K <= 0:
            st.error("Delivery price K must be strictly positive for a Forward.")
        else:
            # -------- Calcul et Stockage --------
            data = compute_linear_product(product_type, S0, r, q, T, K)
            st.session_state.linear_data = data

    # AFFICHAGE DES RÉSULTATS (PERSISTANT)
    if 'linear_data' in st.session_state:
        res = st.session_state.linear_data

        # Petit check de sécurité au cas où l'utilisateur change de type sans recalculer
        # (Optionnel, mais propre pour l'UX : on affiche ce qui a été calculé)

        if res["type"] == "Forward":
            c1, c2 = st.columns(2)
            c1.metric("Forward price F₀", f"{res['F0']:.6f}")
            c2.metric("Forward value V₀", f"{res['V0']:.6f}")

            st.caption(
                "Forward price: F₀ = S₀ · exp((r − q)T). "
                "Value V₀ = exp(−rT) · (F₀ − K)."
            )

        else:
            st.metric("Future price", f"{res['F0']:.6f}")
            st.caption(
                "Under deterministic interest rates, futures and forwards have the same price."
            )
