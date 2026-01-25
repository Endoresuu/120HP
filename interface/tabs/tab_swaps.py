import streamlit as st
from pricer.products.swap import InterestRateSwap


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
        payer = st.radio(
            "Position",
            ["Payer (pay fixed)", "Receiver (receive fixed)"],
            key="swap_pos"
        )

    if st.button("Price swap", key="swap_btn"):

        swap = InterestRateSwap(
            fixed_rate=fixed_rate,
            maturity=T,
            freq=freq,
            notional=notional
        )

        par_rate = swap.par_rate(r)
        value = swap.value(r, payer=payer.startswith("Payer"))

        st.success(f"Par swap rate: {par_rate:.6f}")
        st.info(f"Swap present value: {value:.6f}")
