import numpy as np
from pricer.market.data import MarketData
from pricer.products.base import Option


class MonteCarloModel:
    """
    Monte Carlo sous Black-Scholes.
    """

    def __init__(self, sigma: float, n_paths: int = 10000):
        self.sigma = sigma
        self.n_paths = n_paths

    def price(self, option: Option, market: MarketData) -> float:
        S0 = market.spot
        r = market.r
        T = option.T
        K = option.K

        Z = np.random.normal(size=self.n_paths)

        ST = S0 * np.exp(
            (r - 0.5 * self.sigma**2) * T + self.sigma * np.sqrt(T) * Z
        )

        payoffs = option.payoff(ST)
        return float(np.exp(-r * T) * payoffs.mean())
