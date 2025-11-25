import math
from scipy.stats import norm
from pricer.market.data import MarketData

class BlackScholesModel:
    """
    Modèle de Black-Scholes pour le pricing de call/put européens.
    """

    def __init__(self, market_data: MarketData, sigma: float):
        self.market_data = market_data
        self.sigma = sigma

    def _d1_d2(self, K: float, T: float) -> tuple[float, float]:
        S0 = self.market_data.spot
        r = self.market_data.r
        q = self.market_data.q
        sigma = self.sigma

        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        num = math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T
        den = sigma * math.sqrt(T)
        d1 = num / den
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def price_call(self, K: float, T: float) -> float:
        S0 = self.market_data.spot
        r = self.market_data.r
        q = self.market_data.q
        d1, d2 = self._d1_d2(K, T)
        return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    def price_put(self, K: float, T: float) -> float:
        S0 = self.market_data.spot
        r = self.market_data.r
        q = self.market_data.q
        d1, d2 = self._d1_d2(K, T)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * math.exp(-q * T) * norm.cdf(-d1)
