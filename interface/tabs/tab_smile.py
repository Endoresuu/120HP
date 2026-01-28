import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pricer.calibration.market_calibrator import MarketSmileCalibrator

# --- FONCTION CACHÉE (Calcul & Graphique) ---

@st.cache_data(show_spinner=False)
def compute_smile_cached(ticker, r):
    """
    Instancie le calibrateur, récupère les données et génère le graphique.
    Retourne le DataFrame et la Figure Matplotlib.
    """
    calibrator = MarketSmileCalibrator(ticker, r=float(r))
    df = calibrator.compute_smile()

    # Si pas de données, on renvoie None
    if df is None or df.empty:
        return None, None

    # On génère la figure directement ici pour la mettre en cache
    fig = calibrator.plot_smile()

    return df, fig

# --- FONCTION DE RENDU ---

def render():
    st.subheader("Volatility Smile")

    # Inputs
    ticker_s = st.text_input("Ticker", value="SPY", key="sml_ticker")
    r_s = st.number_input("Risk-free rate r", value=0.04, key="sml_r")

    # BOUTON DE CALCUL
    if st.button("Compute Smile", key="sml_btn"):

        with st.status("Fetching data and calibrating smile...", expanded=True) as status:
            try:
                # Appel de la fonction cachée
                df, fig = compute_smile_cached(ticker_s, r_s)

                if df is None:
                    status.update(label="No options found!", state="error", expanded=False)
                    st.error("No valid options found for this ticker.")
                else:
                    # Stockage dans la session pour persistance
                    st.session_state.smile_data = {
                        "df": df,
                        "fig": fig,
                        "ticker": ticker_s
                    }
                    status.update(label="Smile computed successfully!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Error", state="error", expanded=False)
                st.error(f"Error computing smile: {e}")

    # AFFICHAGE PERSISTANT
    if 'smile_data' in st.session_state:
        data = st.session_state.smile_data

        # Petit check pour vérifier si les données affichées correspondent aux inputs actuels (Optionnel mais UX friendly)
        if data["ticker"] != ticker_s:
            st.caption(f"⚠️ Displaying results for {data['ticker']} (click 'Compute' to update for {ticker_s})")

        st.success("Smile computed successfully!")

        # 1. Tableau de données
        st.dataframe(data["df"], use_container_width=True)

        # 2. Graphique
        if data["fig"]:
            st.pyplot(data["fig"])

        # 3. Bouton de téléchargement (ne recharge plus la page en perdant les données)
        csv = data["df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download smile as CSV",
            csv,
            file_name=f"volatility_smile_{data['ticker']}.csv",
            mime="text/csv"
        )
