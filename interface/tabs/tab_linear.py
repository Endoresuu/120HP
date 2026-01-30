import streamlit as st
from pricer.products.forward import Forward
from pricer.products.future import Future

# =====================================================
# --- FONCTION DE CALCUL CACHÉE ---
# =====================================================

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

    else:  # Future
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

# =====================================================
# --- FONCTION DE RENDU ---
# =====================================================

def render():
    st.subheader("Forwards & Futures")
    st.caption("Pricing of linear products under no-arbitrage assumptions")

    st.divider()

    # =====================================================
    # INPUTS
    # =====================================================
    col_inputs, col_outputs = st.columns([1.3, 1.7], gap="large")

    with col_inputs:
        st.subheader("Market & Contract")

        c1, c2 = st.columns(2)
        with c1:
            S0 = st.number_input(
                "Spot S₀",
                value=100.0,
                min_value=1e-8,
                key="lin_S0"
            )
            r = st.number_input(
                "Risk-free rate r",
                value=0.04,
                key="lin_r"
            )

        with c2:
            q = st.number_input(
                "Dividend yield q",
                value=0.0,
                key="lin_q"
            )
            T = st.number_input(
                "Maturity T (years)",
                value=1.0,
                min_value=1e-6,
                key="lin_T"
            )

        st.markdown("### Product specification")

        product_type = st.radio(
            "Product type",
            ["Forward", "Future"],
            key="lin_type",
            horizontal=True
        )

        K = st.number_input(
            "Delivery price K (Forward only)",
            value=100.0,
            min_value=1e-8,
            disabled=(product_type == "Future"),
            key="lin_K"
        )

        run = st.button(
            "Price linear product",
            use_container_width=True,
            key="lin_btn"
        )

    # =====================================================
    # OUTPUTS
    # =====================================================
    with col_outputs:
        st.subheader("Results")

        if run:
            # -------- Validation (reste dans l'UI) --------
            if S0 <= 0 or T <= 0:
                st.error("Spot S₀ and maturity T must be strictly positive.")
            elif product_type == "Forward" and K <= 0:
                st.error("Delivery price K must be strictly positive for a Forward.")
            else:
                # -------- Calcul et Stockage --------
                st.session_state.linear_data = compute_linear_product(
                    product_type, S0, r, q, T, K
                )

        # AFFICHAGE DES RÉSULTATS (PERSISTANT)
        if "linear_data" in st.session_state:
            res = st.session_state.linear_data

            # Petit check UX : on affiche ce qui a été effectivement calculé
            if res["type"] == "Forward":
                m1, m2 = st.columns(2)
                m1.metric("Forward price F₀", f"{res['F0']:.6f}")
                m2.metric("Forward value V₀", f"{res['V0']:.6f}")

                st.caption(
                    "Forward price: F₀ = S₀ · exp((r − q)T). "
                    "Value V₀ = exp(−rT) · (F₀ − K)."
                )

            else:
                st.metric("Future price", f"{res['F0']:.6f}")
                st.caption(
                    "Under deterministic interest rates, futures and forwards "
                    "have the same price."
                )
