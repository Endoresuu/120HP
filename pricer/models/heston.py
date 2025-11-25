import numpy as np
from pricer.market.data import MarketData
from pricer.products.base import Option


class HestonModel:
    """
    Modèle de Heston avec simulation de trajectoires par schéma d'Euler
    et pricing Monte Carlo d'une option européenne.
    On suppose que l'objet MarketData contient tous les paramètres nécessaires.
    """

    def __init__(self):
        # Pas besoin d'hériter de Model pour l'instant
        pass

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

        T = market.T
        n_steps = market.n_steps
        n_paths = market.n_paths

        dt = T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = market.S_0
        v[:, 0] = market.v_0

        sqrt_dt = np.sqrt(dt)
        sqrt_1_minus_rho2 = np.sqrt(1.0 - market.rho**2)

        for t in range(n_steps):
            z1 = np.random.normal(size=n_paths)
            z2 = np.random.normal(size=n_paths)

            dW1 = sqrt_dt * z1
            dW2 = sqrt_dt * (market.rho * z1 + sqrt_1_minus_rho2 * z2)

            v_t = np.maximum(v[:, t], 0.0)

            v_next = (
                v_t
                + market.kappa * (market.theta - v_t) * dt
                + market.sigma_v * np.sqrt(v_t) * dW2
            )
            v[:, t + 1] = np.maximum(v_next, 0.0)

            S[:, t + 1] = S[:, t] * np.exp(
                (market.r - 0.5 * v_t) * dt + np.sqrt(v_t) * dW1
            )

        return S, v

    def price_european(self, option: Option, market: MarketData) -> float:
        """
        Prix Monte Carlo d'une option européenne (call/put)
        dont le payoff est donné par option.payoff(ST).
        """
        S, _ = self.simulate_paths(market)
        ST = S[:, -1]

        payoffs = np.array([option.payoff(s) for s in ST])
        discount_factor = np.exp(-market.r * option.T)

        price = discount_factor * payoffs.mean()
        return float(price)
