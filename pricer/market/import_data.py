import yfinance as yf
import pandas as pd

def get_info(name: str) -> dict:
    tick = yf.Ticker(name)
    return tick.info


def get_close_price(name: str, start=None, end=None) -> pd.Series:
    tick = yf.Ticker(name)

    if start is None and end is None:
        data = tick.history(period="max")
    else:
        data = tick.history(start=start, end=end)

    return data["Close"]


def get_option_chain(name: str) -> dict:
    """
    Récupère les chaînes d'options Yahoo Finance,
    et sélectionne automatiquement les colonnes disponibles.
    """
    tick = yf.Ticker(name)
    expiries = tick.options
    chains = {}

    wanted = ["strike", "lastPrice", "bid", "ask", "impliedVol", "impliedVolatility"]

    for exp in expiries:
        chain = tick.option_chain(exp)
        calls = chain.calls

        available = [col for col in wanted if col in calls.columns]
        chains[exp] = calls[available]

    return chains
