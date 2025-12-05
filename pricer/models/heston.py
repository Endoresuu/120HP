import numpy as np
from pricer.market.data import MarketData
from pricer.products.base import Option


class HestonModel:
    """
    Modèle de Heston + simulation Euler + pricing Monte Carlo.
    """

    def simulate_paths(self, market: MarketData):
        T = market.T
        n_steps = market.n_steps
        n_paths = market.n_paths

        dt = T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = market.S_0
        v[:, 0] = market.v_0

        sqrt_dt = np.sqrt(dt)
        rho = market.rho
        kappa = market.kappa
        theta = market.theta
        sigma_v = market.sigma_v

        for t in range(n_steps):
            z1 = np.random.normal(size=n_paths)
            z2 = np.random.normal(size=n_paths)

            dW1 = sqrt_dt * z1
            dW2 = sqrt_dt * (rho * z1 + np.sqrt(1 - rho**2) * z2)

            v_t = np.maximum(v[:, t], 0)

            # CIR variance process
            v_next = (
                v_t
                + kappa * (theta - v_t) * dt
                + sigma_v * np.sqrt(v_t) * dW2
            )
            v[:, t + 1] = np.maximum(v_next, 0)

            # Asset dynamics
            S[:, t + 1] = S[:, t] * np.exp(
                (market.r - 0.5 * v_t) * dt + np.sqrt(v_t) * dW1
            )

        return S, v

    def price_european(self, option: Option, market: MarketData):
        S, _ = self.simulate_paths(market)
        ST = S[:, -1]

        payoffs = option.payoff(ST)
        price = np.exp(-market.r * option.T) * payoffs.mean()

        return float(price)
