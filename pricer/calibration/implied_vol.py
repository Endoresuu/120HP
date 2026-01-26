import numpy as np
from scipy.stats import norm

class _BlackScholesForIV:
    """
    Minimal Black-Scholes model implementation optimized for Implied Volatility (IV) inversion.

    This class provides the core mathematical functions (Price and Vega) required
    by root-finding algorithms to back out volatility from market prices.

    Methods:
        call_price: Calculates the European Call option price.
        vega: Calculates the option's sensitivity to volatility.
    """
    def call_price(self, S0, K, T, r, sigma):
        """
        Calculates the theoretical price of a European Call option.

        Args:
            S0 (float): Current price of the underlying asset.
            K (float): Strike price (exercise price).
            T (float): Time to maturity in years.
            r (float): Annualized risk-free interest rate.
            sigma (float): Annualized volatility of the underlying asset.

        Returns:
            float: The Black-Scholes European Call option price.
        """
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def vega(self, S0, K, T, r, sigma):
        """
        Calculates the Vega of a European option.

        Vega measures the rate of change of the option price with respect to
        the volatility of the underlying asset. It is the first derivative
        of the price function relative to sigma.

        Args:
            S0 (float): Current price of the underlying asset.
            K (float): Strike price.
            T (float): Time to maturity in years.
            r (float): Annualized risk-free interest rate.
            sigma (float): Annualized volatility.

        Returns:
            float: The Vega of the option (change in price for a 1% change in sigma).
        """
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S0 * norm.pdf(d1) * np.sqrt(T)


class NewtonImpliedVolSolver:
    """
    A robust solver for calculating Implied Volatility using a hybrid Newton-Raphson method.

    This solver finds the volatility (sigma) that matches the Black-Scholes model price
    to the market price. It uses a bracketing technique to fall back on bisection
    if Newton-Raphson moves outside of valid boundaries, ensuring convergence.

    Attributes:
        tol (float): Convergence tolerance for the price difference.
        max_iter (int): Maximum number of iterations allowed.
        sigma_min (float): Minimum volatility floor to prevent numerical instability.
        sigma_max_cap (float): Maximum volatility ceiling.
        model (_BlackScholesForIV): The underlying pricing engine.
    """

    def __init__(self,
                 tol=1e-6,
                 max_iter=100,
                 sigma_min=1e-4,
                 sigma_max_cap=5.0):
        """
        Initializes the solver with convergence parameters.
        """
        self.tol = tol
        self.max_iter = max_iter
        self.sigma_min = sigma_min
        self.sigma_max_cap = sigma_max_cap
        self.model = _BlackScholesForIV()

    def _initial_guess(self, S0, K, T, r, C_mkt):
        """
        Provides a starting volatility value using the Brenner-Subrahmanyam approximation.

        Formula used: sigma approx sqrt(2*pi / T) * (C / S0)
        """
        sigma0 = np.sqrt(2 * np.pi / T) * (C_mkt / S0)
        return float(np.clip(sigma0, 0.05, 1.0))

    def _find_bracket(self, S0, K, T, r, C_mkt):
        """
        Identifies an interval [a, b] where the true volatility must reside.

        This is used to constrain the Newton-Raphson steps and provide a fallback
        for bisection.

        Returns:
            tuple: (lower_bound, upper_bound) or (None, None) if no bracket is found.
        """
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
        """
        Executes the iterative solver to find the Implied Volatility.

        The method performs a check for arbitrage bounds (intrinsic value) before
        starting. It then iterates using Newton-Raphson:
        sigma_{n+1} = sigma_n - (C(sigma_n) - C_market) / Vega(sigma_n).

        Args:
            option (Object): An object containing S0, K, T, r, and price_mkt.
            sigma0 (float, optional): Initial guess for volatility.
            track (bool): If True, returns the history of sigma and error values.

        Returns:
            float or tuple: The calculated IV, or (IV, sigma_history, error_history) if track=True.
                           Returns np.nan if the solution fails to converge or inputs are invalid.
        """
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
