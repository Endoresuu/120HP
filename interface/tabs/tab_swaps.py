import streamlit as st
from pricer.products.swap import InterestRateSwap

# --- FONCTION DE CALCUL CACHÉE ---
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

# --- FONCTION DE RENDU ---
def render():

    st.header("Interest Rate Swap (Fixed vs Floating)")

    col1, col2 = st.columns(2)

    with col1:
        r = st.number_input("Flat interest rate r", value=0.03, key="swap_r")
        T = st.number_input("Maturity T (years)", value=5.0, key="swap_T")
        fixed_rate = st.number_input("Fixed rate K", value=0.03, key="swap_K")

    with col2:
        notional = st.number_input("Notional", value=1.0, key="swap_N")
        freq = st.selectbox("Payment frequency (per year)", [1, 2, 4], key="swap_freq")
        payer_selection = st.radio(
            "Position",
            ["Payer (pay fixed)", "Receiver (receive fixed)"],
            key="swap_pos"
        )

    # BOUTON DE CALCUL
    if st.button("Price swap", key="swap_btn"):

        # Préparation du booléen pour la fonction cachée
        is_payer = payer_selection.startswith("Payer")

        # Calcul et Stockage
        results = compute_swap_cached(
            r=r,
            T=T,
            fixed_rate=fixed_rate,
            notional=notional,
            freq=freq,
            is_payer=is_payer
        )
        st.session_state.swap_results = results

    # AFFICHAGE PERSISTANT
    if 'swap_results' in st.session_state:
        res = st.session_state.swap_results

        st.divider()
        st.success(f"Par swap rate: {res['par_rate']:.6f}")

        # On peut ajouter une petite nuance visuelle selon si la valeur est positive ou négative
        val = res['value']
        lbl = "Swap Present Value"
        if val >= 0:
            st.info(f"{lbl}: {val:.6f}")
        else:
            st.error(f"{lbl}: {val:.6f}")

        # Petit rappel contextuel
        direction = "paying" if res['is_payer'] else "receiving"
        st.caption(f"Value for the party {direction} fixed rate.")
