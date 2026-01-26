from datetime import datetime
import pandas as pd


def choose_next_expiry(chains):
    """
    Retourne (expiry_str, days, T) pour la prochaine échéance future.
    """
    if not chains:
        return None, None, None

    expiries = sorted(chains.keys())
    today = datetime.today().date()

    best_expiry = None
    best_days = None

    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue

        days = (d - today).days
        if days <= 0:
            continue

        if best_days is None or days < best_days:
            best_days = days
            best_expiry = e

    if best_expiry is None:
        return None, None, None

    return best_expiry, best_days, best_days / 365.0


def choose_expiry_closest_to_T(chains, T_target):
    """
    Retourne (expiry_str, days, T) pour l'échéance la plus proche de T_target.
    """
    if not chains or T_target is None or T_target <= 0:
        return None, None, None

    expiries = sorted(chains.keys())
    today = datetime.today().date()

    best_expiry = None
    best_days = None
    best_diff = None

    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue

        days = (d - today).days
        if days <= 0:
            continue

        T = days / 365.0
        diff = abs(T - T_target)

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_expiry = e
            best_days = days

    if best_expiry is None:
        return None, None, None

    return best_expiry, best_days, best_days / 365.0


def get_market_call_price_from_chain(chains, expiry, K):
    """
    Récupère un prix de CALL depuis une option chain Yahoo Finance
    en prenant le strike le plus proche de K.
    """
    if chains is None or expiry not in chains:
        return None, None

    df = chains.get(expiry)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None

    if "strike" not in df.columns:
        return None, None

    df = df.sort_values("strike")
    idx = (df["strike"] - K).abs().idxmin()
    row = df.loc[idx]

    price = None

    if "lastPrice" in df.columns and pd.notna(row.get("lastPrice")) and row["lastPrice"] > 0:
        price = float(row["lastPrice"])

    elif "bid" in df.columns and "ask" in df.columns:
        bid = row.get("bid")
        ask = row.get("ask")
        if bid is not None and ask is not None and ask >= bid:
            price = float(0.5 * (bid + ask))

    return price, float(row["strike"])
