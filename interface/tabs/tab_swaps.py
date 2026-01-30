import streamlit as st
from pricer.products.swap import InterestRateSwap

# =====================================================
# --- FONCTION DE CALCUL CACHÉE ---
# =====================================================

@st.cache_data
def compute_swap_cached(r, T, fixed_rate, notional, freq, is_payer):
    """
    Effectue les calculs du Swap.
    """
    swap = InterestRateSwap(
        fixed_rate=fixed_rate,
        maturity=T,
        freq=freq,
        notional=notional
    )

    par_rate = swap.par_rate(r)
    value = swap.value(r, payer=is_payer)

    return {
        "par_rate": par_rate,
        "value": value,
        "is_payer": is_payer
    }

# =====================================================
# --- FONCTION DE RENDU ---
# =====================================================

def render():
    st.subheader("Interest Rate Swap (Fixed vs Floating)")
    st.caption("Plain-vanilla fixed-for-floating interest rate swap pricing")

    st.divider()

    # =====================================================
    # INPUTS
    # =====================================================
    col_inputs, col_outputs = st.columns([1.3, 1.7], gap="large")

    with col_inputs:
        st.subheader("Swap parameters")

        c1, c2 = st.columns(2)
        with c1:
            r = st.number_input(
                "Flat interest rate r",
                value=0.03,
                key="swap_r"
            )
            T = st.number_input(
                "Maturity T (years)",
                value=5.0,
                key="swap_T"
            )

        with c2:
            fixed_rate = st.number_input(
                "Fixed rate K",
                value=0.03,
                key="swap_K"
            )
            freq = st.selectbox(
                "Payment frequency (per year)",
                [1, 2, 4],
                key="swap_freq"
            )

        notional = st.number_input(
            "Notional",
            value=1.0,
            key="swap_N"
        )

        payer_selection = st.radio(
            "Position",
            ["Payer (pay fixed)", "Receiver (receive fixed)"],
            key="swap_pos",
            horizontal=True
        )

        run = st.button(
            "Price swap",
            use_container_width=True,
            key="swap_btn"
        )

    # =====================================================
    # OUTPUTS
    # =====================================================
    with col_outputs:
        st.subheader("Results")

        if run:
            # Préparation du booléen pour la fonction cachée
            is_payer = payer_selection.startswith("Payer")

            # Calcul et stockage
            st.session_state.swap_results = compute_swap_cached(
                r=r,
                T=T,
                fixed_rate=fixed_rate,
                notional=notional,
                freq=freq,
                is_payer=is_payer
            )

        # AFFICHAGE PERSISTANT
        if "swap_results" in st.session_state:
            res = st.session_state.swap_results

            st.metric("Par swap rate", f"{res['par_rate']:.6f}")

            val = res["value"]
            label = "Swap Present Value"

            if val >= 0:
                st.info(f"{label}: {val:.6f}")
            else:
                st.error(f"{label}: {val:.6f}")

            direction = "paying" if res["is_payer"] else "receiving"
            st.caption(f"Value for the party {direction} fixed rate.")
