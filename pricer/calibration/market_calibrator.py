import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from pricer.products.market_option import MarketOption
from pricer.calibration.implied_vol import NewtonImpliedVolSolver
from pricer.market.import_data import get_option_chain, get_close_price


class MarketSmileCalibrator:
    """
    Calibrator of the implied volatility smile for a single maturity.
    """

    def __init__(self, ticker, r=0.04,
                 min_maturity_days=15, max_maturity_days=45,
                 min_price=0.10,
                 moneyness_cutoff=0.5,
                 tol_rel_increase=0.25):
        """
        Initializes the calibrator with market filtering and maturity selection rules.
        """
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
        """
        Downloads option chains and stock price, selecting the nearest valid maturity.

        The method scans available expiration dates and selects the first one
        falling within the [min_days, max_days] window.

        Raises:
            RuntimeError: If no maturity is found within the specified day range.
        """
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
        """
        Processes the option chain to compute implied volatilities for valid strikes.

        This method performs several data cleaning steps:
        1. Filters out zero-volume or illiquid options.
        2. Validates Bid/Ask consistency.
        3. Enforces the Lower Arbitrage Bound (C_mkt >= S0 - K*exp(-rT)).
        4. Removes strikes that violate the monotonicity of call prices.
        5. Solves for IV using the NewtonImpliedVolSolver.

        Returns:
            pd.DataFrame: A DataFrame containing 'strike' and 'iv' columns.
        """
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

    def plot_smile(self):
        """
        Generates a visualization of the volatility smile.

        Returns:
            matplotlib.figure.Figure: The figure object showing IV vs Strike.

        Raises:
            RuntimeError: If called before compute_smile() or if no data is available.
        """
        if self.df_result is None or self.df_result.empty:
            raise RuntimeError("No smile data to plot.")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.df_result["strike"], self.df_result["iv"], marker="o")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied volatility")
        ax.set_title(f"Volatility Smile – {self.ticker} – {self.expiry}")
        ax.grid(True)

        return fig
