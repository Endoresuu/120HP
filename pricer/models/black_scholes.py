import math
from scipy.stats import norm
from pricer.market.data import MarketData

class BlackScholesModel:
    """
    Standard Black-Scholes-Merton model for European option pricing.

    This class implements the analytical formulas for European Call and Put options,
    accounting for continuous dividend yields and risk-free rates.

    Attributes:
        market_data (MarketData): Container for market observables (S0, r, q).
        sigma (float): Annualized constant volatility.
    """
    def __init__(self, market_data: MarketData, sigma: float):
        """
        Initializes the model with market conditions and a specific volatility.

        Args:
            market_data (MarketData): The environment parameters.
            sigma (float): The volatility parameter used for pricing.
        """
        self.market_data = market_data
        self.sigma = sigma

    def _d1_d2(self, K: float, T: float) -> tuple[float, float]:
        """
        Calculates the probability-weighted factors d1 and d2.

        These intermediate variables represent the moneyness and probability
        of exercise in the risk-neutral measure.

        Args:
            K (float): Strike price.
            T (float): Time to maturity in years.

        Returns:
            tuple[float, float]: (d1, d2)

        Raises:
            ValueError: If T is not strictly positive, as the formula contains sqrt(T).
        """
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
        """
        Calculates the price of a European Put option.

        Uses the formula:
        P = K * e^(-rT) * N(-d2) - S0 * e^(-qT) * N(-d1)

        Args:
            K (float): Strike price.
            T (float): Time to maturity.

        Returns:
            float: Theoretical Put price.
        """
        S0 = self.market_data.spot
        r = self.market_data.r
        q = self.market_data.q
        d1, d2 = self._d1_d2(K, T)
        return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    def price_put(self, K: float, T: float) -> float:
        """
        Calculates the price of a European Call option.

        Uses the formula:
        C = S0 * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)

        Args:
            K (float): Strike price.
            T (float): Time to maturity.

        Returns:
            float: Theoretical Call price.
        """
        S0 = self.market_data.spot
        r = self.market_data.r
        q = self.market_data.q
        d1, d2 = self._d1_d2(K, T)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * math.exp(-q * T) * norm.cdf(-d1)
