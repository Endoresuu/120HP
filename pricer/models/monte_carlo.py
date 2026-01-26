import numpy as np
from pricer.market.data import MarketData
from pricer.products.base import Option


class MonteCarloModel:
    """
    Standard Monte Carlo pricing engine for European options under Geometric Brownian Motion.

    This model simulates the asset price at maturity (T) by generating random
    standard normal samples and applying them to the risk-neutral price process.
    It is the numerical counterpart to the analytical Black-Scholes model.

    Attributes:
        sigma (float): Constant annualized volatility.
        n_paths (int): Number of simulated price trajectories (sample size).
    """
    def __init__(self, sigma: float, n_paths: int = 10000):
        """
        Initializes the Monte Carlo engine with volatility and simulation size.
        """
        self.sigma = sigma
        self.n_paths = n_paths

    def price(self, option: Option, market: MarketData) -> float:
        """
        Calculates the option price by averaging discounted simulated payoffs.

        The method follows these steps:
        1. Generate N random draws from a standard normal distribution.
        2. Simulate terminal stock prices (ST) using the solution to the SDE.
        3. Compute the payoff for each path based on the option type.
        4. Discount the average payoff back to the present value at rate 'r'.

        Args:
            option (Option): An object defining the contract (K, T, and payoff logic).
            market (MarketData): Container for current market parameters (spot, r).

        Returns:
            float: The estimated fair value of the option.
        """
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
