import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from pricer.products.market_option import MarketOption
from pricer.calibration.implied_vol import NewtonImpliedVolSolver
from pricer.market.import_data import get_option_chain, get_close_price


class MarketSmileCalibrator:
    """
    Calibrateur du smile de volatilité implicite pour une seule maturité.
    """

    def __init__(self, ticker, r=0.04,
                 min_maturity_days=15, max_maturity_days=45,
                 min_price=0.10,
                 moneyness_cutoff=0.5,
                 tol_rel_increase=0.25):
        
        self.ticker = ticker
        self.r = r
        self.min_days = min_maturity_days
        self.max_days = max_maturity_days
        self.min_price = min_price
        self.moneyness_cutoff = moneyness_cutoff
        self.tol_rel_increase = tol_rel_increase

        self.S0 = None
        self.expiry = None
        self.T = None
        self.df_calls = None
        self.df_result = None

        self.solver = NewtonImpliedVolSolver()

    # -----------------------------------------------------------
    # 1. Télécharger et préparer les données
    # -----------------------------------------------------------
    def load_market_data(self):
        chains = get_option_chain(self.ticker)
        self.S0 = float(get_close_price(self.ticker).iloc[-1])

        expiries_sorted = sorted(chains.keys())
        chosen = None

        for e in expiries_sorted:
            days = (datetime.strptime(e, "%Y-%m-%d") - datetime.today()).days
            if self.min_days <= days <= self.max_days:
                chosen = e
                break

        if chosen is None:
            raise RuntimeError("Aucune maturité dans la fenêtre demandée.")

        self.expiry = chosen
        self.T = days / 365.0

        df = chains[self.expiry].copy()
        df = df.sort_values("strike").reset_index(drop=True)
        self.df_calls = df

    # -----------------------------------------------------------
    # 2. Calcul du smile
    # -----------------------------------------------------------
    def compute_smile(self):

        if self.df_calls is None:
            self.load_market_data()

        S0 = self.S0
        r = self.r
        T = self.T

        strikes = []
        vols = []

        prev_price = None

        for _, row in self.df_calls.iterrows():

            # Volume 0 → on skip
            if "volume" in self.df_calls.columns:
                v = row["volume"]
                if pd.notna(v) and v == 0:
                    continue

            # bid/ask inversé
            if "bid" in row and "ask" in row:
                bid, ask = row["bid"], row["ask"]
                if pd.notna(bid) and pd.notna(ask) and ask < bid:
                    continue

            K = float(row["strike"])
            C_mkt = float(row.get("lastPrice", np.nan))

            if not np.isfinite(C_mkt) or C_mkt <= 0:
                continue

            if abs(np.log(K / S0)) > self.moneyness_cutoff:
                continue

            if C_mkt < self.min_price:
                continue

            lower = max(0.0, S0 - K*np.exp(-r*T))
            if C_mkt < lower - 1e-4:
                continue

            if prev_price is not None:
                if C_mkt > prev_price * (1 + self.tol_rel_increase):
                    continue

            prev_price = C_mkt

            opt = MarketOption(S0, K, T, r, C_mkt)
            sigma = self.solver.solve(opt)

            if not np.isfinite(sigma) or not (0.01 <= sigma <= 2.0):
                continue

            strikes.append(K)
            vols.append(sigma)

        if not strikes:
            self.df_result = pd.DataFrame(columns=["strike", "iv"])
        else:
            self.df_result = pd.DataFrame({
                "strike": strikes,
                "iv": vols
            }).sort_values("strike")

        return self.df_result

    # -----------------------------------------------------------
    # 3. Plot
    # -----------------------------------------------------------
    def plot_smile(self, filename=None):

        if self.df_result is None or self.df_result.empty:
            raise RuntimeError("Aucun résultat pour tracer le smile.")

        plt.figure(figsize=(8, 5))
        plt.plot(self.df_result["strike"], self.df_result["iv"], marker="o")
        plt.grid(True)
        plt.xlabel("Strike")
        plt.ylabel("Vol implicite")
        plt.title(f"Smile – {self.ticker} – {self.expiry}")

        if filename:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()
