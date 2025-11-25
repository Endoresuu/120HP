import numpy as np
from scipy.stats import norm

class _BlackScholesForIV:
    """
    Modèle BS minimal utilisé uniquement pour l'inversion de volatilité implicite.
    """
    def call_price(self, S0, K, T, r, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def vega(self, S0, K, T, r, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S0 * norm.pdf(d1) * np.sqrt(T)

class NewtonImpliedVolSolver:

    def __init__(self,
                 tol=1e-6,
                 max_iter=100,
                 sigma_min=1e-4,
                 sigma_max_cap=5.0):
        self.tol = tol
        self.max_iter = max_iter
        self.sigma_min = sigma_min
        self.sigma_max_cap = sigma_max_cap
        self.model = _BlackScholesForIV()

    def _initial_guess(self, S0, K, T, r, C_mkt):
        sigma0 = np.sqrt(2 * np.pi / T) * (C_mkt / S0)
        return float(np.clip(sigma0, 0.05, 1.0))

    def _find_bracket(self, S0, K, T, r, C_mkt):
        a = self.sigma_min
        C_a = self.model.call_price(S0, K, T, r, a)

        if C_a > C_mkt:
            return None, None

        b = 0.5
        C_b = self.model.call_price(S0, K, T, r, b)

        while C_b < C_mkt and b < self.sigma_max_cap:
            b *= 2
            C_b = self.model.call_price(S0, K, T, r, b)

        if C_b < C_mkt:
            return None, None

        return a, b

    def solve(self, option, sigma0=None, track=False):
        S0, K, T, r, C_mkt = option.S0, option.K, option.T, option.r, option.price_mkt

        if T <= 0:
            return np.nan
        if C_mkt <= 0 or C_mkt >= S0:
            return np.nan

        intrinsic = max(0, S0 - K * np.exp(-r * T))
        if C_mkt < intrinsic:
            return np.nan

        a, b = self._find_bracket(S0, K, T, r, C_mkt)
        if a is None:
            return np.nan

        sigma = sigma0 or self._initial_guess(S0, K, T, r, C_mkt)
        sigma = float(np.clip(sigma, a, b))

        sig_hist, err_hist = [], []

        for _ in range(self.max_iter):
            C_bs = self.model.call_price(S0, K, T, r, sigma)
            vega = self.model.vega(S0, K, T, r, sigma)
            diff = C_bs - C_mkt

class NewtonImpliedVolSolver:

    def __init__(self,
                 tol=1e-6,
                 max_iter=100,
                 sigma_min=1e-4,
                 sigma_max_cap=5.0):
        self.tol = tol
        self.max_iter = max_iter
        self.sigma_min = sigma_min
        self.sigma_max_cap = sigma_max_cap
        self.model = _BlackScholesForIV()

    def _initial_guess(self, S0, K, T, r, C_mkt):
        sigma0 = np.sqrt(2 * np.pi / T) * (C_mkt / S0)
        return float(np.clip(sigma0, 0.05, 1.0))

    def _find_bracket(self, S0, K, T, r, C_mkt):
        a = self.sigma_min
        C_a = self.model.call_price(S0, K, T, r, a)

        if C_a > C_mkt:
            return None, None

        b = 0.5
        C_b = self.model.call_price(S0, K, T, r, b)

        while C_b < C_mkt and b < self.sigma_max_cap:
            b *= 2
            C_b = self.model.call_price(S0, K, T, r, b)

        if C_b < C_mkt:
            return None, None

        return a, b

    def solve(self, option, sigma0=None, track=False):
        S0, K, T, r, C_mkt = option.S0, option.K, option.T, option.r, option.price_mkt

        if T <= 0:
            return np.nan
        if C_mkt <= 0 or C_mkt >= S0:
            return np.nan

        intrinsic = max(0, S0 - K * np.exp(-r * T))
        if C_mkt < intrinsic:
            return np.nan

        a, b = self._find_bracket(S0, K, T, r, C_mkt)
        if a is None:
            return np.nan

        sigma = sigma0 or self._initial_guess(S0, K, T, r, C_mkt)
        sigma = float(np.clip(sigma, a, b))

        sig_hist, err_hist = [], []

        for _ in range(self.max_iter):
            C_bs = self.model.call_price(S0, K, T, r, sigma)
            vega = self.model.vega(S0, K, T, r, sigma)
            diff = C_bs - C_mkt

            if track:
                sig_hist.append(sigma)
                err_hist.append(diff)

            if abs(diff) < self.tol:
                return (sigma, sig_hist, err_hist) if track else sigma

            if vega > 1e-8:
                sigma_new = sigma - diff / vega
            else:
                sigma_new = None

            if sigma_new is not None and a <= sigma_new <= b:
                sigma = sigma_new
            else:
                C_a = self.model.call_price(S0, K, T, r, a) - C_mkt
                if C_a * diff < 0:
                    b = sigma
                else:
                    a = sigma
                sigma = 0.5 * (a + b)

            sigma = float(np.clip(sigma, self.sigma_min, self.sigma_max_cap))

        return (np.nan, sig_hist, err_hist) if track else np.nan
