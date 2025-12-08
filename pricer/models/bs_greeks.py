import numpy as np
from scipy.stats import norm

def d1(S, K, r, sigma, T):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma*np.sqrt(T)


def bs_call_price(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return S * norm.cdf(D1) - K * np.exp(-r*T) * norm.cdf(D2)


def delta_call(S, K, r, sigma, T):
    return norm.cdf(d1(S, K, r, sigma, T))

def delta_put(S, K, r, sigma, T):
    return norm.cdf(d1(S, K, r, sigma, T)) - 1.0


def gamma(S, K, r, sigma, T):
    return norm.pdf(d1(S, K, r, sigma, T)) / (S * sigma * np.sqrt(T))


def vega(S, K, r, sigma, T):
    return S * norm.pdf(d1(S, K, r, sigma, T)) * np.sqrt(T)


def theta_call(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    term1 = - (S * norm.pdf(D1) * sigma) / (2*np.sqrt(T))
    term2 = - r * K * np.exp(-r*T) * norm.cdf(D2)
    return term1 + term2


def theta_put(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    term1 = - (S * norm.pdf(D1) * sigma) / (2*np.sqrt(T))
    term2 = + r * K * np.exp(-r*T) * norm.cdf(-D2)
    return term1 + term2


def rho_call(S, K, r, sigma, T):
    return K * T * np.exp(-r*T) * norm.cdf(d2(S, K, r, sigma, T))

def rho_put(S, K, r, sigma, T):
    return -K * T * np.exp(-r*T) * norm.cdf(-d2(S, K, r, sigma, T))
