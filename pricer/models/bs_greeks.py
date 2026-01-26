import numpy as np
from scipy.stats import norm

def d1(S, K, r, sigma, T):
    """
    Calculates the d1 component of the Black-Scholes formula.

    d1 represents the distance to the strike price in standard deviations
    under the risk-neutral measure, adjusted for the drift.
    """
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, sigma, T):
    """
    Calculates the d2 component of the Black-Scholes formula.

    d2 is related to the probability that the option will be exercised
    at maturity (N(d2)).
    """
    return d1(S, K, r, sigma, T) - sigma*np.sqrt(T)


def bs_call_price(S, K, r, sigma, T):
    """
    Computes the Black-Scholes price for a European Call option.
    """
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return S * norm.cdf(D1) - K * np.exp(-r*T) * norm.cdf(D2)


def delta_call(S, K, r, sigma, T):
    """
    Calculates the Delta of a Call option.

    Delta measures the rate of change of the option price with respect
    to changes in the underlying asset's price.
    """
    return norm.cdf(d1(S, K, r, sigma, T))

def delta_put(S, K, r, sigma, T):
    """
    Calculates the Delta of a Put option.

    Ranges from -1 to 0. It represents the sensitivity of the Put
    to the underlying price.
    """
    return norm.cdf(d1(S, K, r, sigma, T)) - 1.0


def gamma(S, K, r, sigma, T):
    """
    Calculates the Gamma (same for Call and Put).

    Gamma measures the rate of change in Delta per 1-unit move in the
    underlying price. It represents the 'convexity' of the option.
    """
    return norm.pdf(d1(S, K, r, sigma, T)) / (S * sigma * np.sqrt(T))


def vega(S, K, r, sigma, T):
    """
    Calculates the Vega (same for Call and Put).

    Vega measures sensitivity to volatility. It is the change in
    option price for a 1% change in sigma.
    """
    return S * norm.pdf(d1(S, K, r, sigma, T)) * np.sqrt(T)


def theta_call(S, K, r, sigma, T):
    """
    Calculates the Theta of a Call option.

    Theta measures the 'time decay' or the change in option price
    as time to maturity decreases.
    """
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    term1 = - (S * norm.pdf(D1) * sigma) / (2*np.sqrt(T))
    term2 = - r * K * np.exp(-r*T) * norm.cdf(D2)
    return term1 + term2


def theta_put(S, K, r, sigma, T):
    """
    Calculates the Theta of a Put option.
    """
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    term1 = - (S * norm.pdf(D1) * sigma) / (2*np.sqrt(T))
    term2 = + r * K * np.exp(-r*T) * norm.cdf(-D2)
    return term1 + term2


def rho_call(S, K, r, sigma, T):
    """
    Calculates the Rho of a Call option.

    Rho measures sensitivity to changes in the risk-free interest rate.
    """
    return K * T * np.exp(-r*T) * norm.cdf(d2(S, K, r, sigma, T))

def rho_put(S, K, r, sigma, T):
    """
    Calculates the Rho of a Put option.
    """
    return -K * T * np.exp(-r*T) * norm.cdf(-d2(S, K, r, sigma, T))
