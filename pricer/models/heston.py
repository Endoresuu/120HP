import numpy as np
import os
import sys
from pricer.market.data import MarketData
from pricer.products.base import Option

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class HestonModel:
    """
    Modèle de Heston + simulation Euler + pricing Monte Carlo.
    """

    def __init__(self, T, K, n_steps=252, n_paths=50000, v0=0.04, kappa=2, theta=0.04, sigma_v=0.5, rho=-0.7):

        self.T = T
        self.K = K
        self. n_steps = n_steps
        self.n_paths = n_paths
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def simulate_paths(self, market: MarketData):
        """
        Simule des trajectoires (S_t, v_t) sous Heston.

        Hypothèses sur market :
        - market.T        : maturité
        - market.n_steps  : nombre de pas de temps
        - market.n_paths  : nombre de trajectoires
        - market.S_0      : spot initial
        - market.v_0      : variance initiale
        - market.r        : taux sans risque
        - market.kappa    : vitesse de rappel
        - market.theta    : variance de long terme
        - market.sigma_v  : volatilité de la variance
        - market.rho      : corrélation entre dW1 et dW2
        """


        dt = self.T / self.n_steps

        S = np.zeros((self.n_paths, self.n_steps + 1))
        v = np.zeros((self.n_paths, self.n_steps + 1))

        S[:, 0] = market.S_0
        v[:, 0] = self.v0

        sqrt_dt = np.sqrt(dt)
        sqrt_1_minus_rho2 = np.sqrt(1.0 - self.rho**2)

        for t in range(self.n_steps):
            z1 = np.random.normal(size=self.n_paths)
            z2 = np.random.normal(size=self.n_paths)

            dW1 = sqrt_dt * z1
            dW2 = sqrt_dt * (self.rho * z1 + sqrt_1_minus_rho2 * z2)

            v_t = np.maximum(v[:, t], 0)

            # CIR variance process
            v_next = (
                v_t
                + self.kappa * (self.theta - v_t) * dt
                + self.sigma_v * np.sqrt(v_t) * dW2
            )
            v[:, t + 1] = np.maximum(v_next, 0)

            # Asset dynamics
            S[:, t + 1] = S[:, t] * np.exp(
                (market.r - 0.5 * v_t) * dt + np.sqrt(v_t) * dW1
            )

        return S, v

    def heston_european_call_mc(self, market: MarketData):

        S, v = self.simulate_paths(market)

        ST = S[:, -1]
        payoff = np.maximum(ST - self.K, 0.0)
        price = np.exp(-market.r * self.T) * payoff.mean()

        return price

    def heston_european_put_mc(self, mearket: MarketData):

        S, v = self.simulate_paths(market)

        ST = S[:, -1]


        payoff = np.maximum(self.K - ST, 0.0)

        price = np.exp(-market.r * self.T) * payoff.mean()


        return price



market = MarketData(500)

heston = HestonModel(1, 100)

print(heston.heston_european_call_mc(market))

print(heston.heston_european_put_mc(market))
