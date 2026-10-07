"""Closed-form prices under Black-Scholes, used as references for the Monte Carlo estimators."""

import numpy as np
from scipy.stats import norm


def _d1_d2(S0, K, T, r, sigma):
    """d1 (= d+) and d2 (= d-), shared by the call and the put."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S0, K, T, r, sigma):
    """European call: C = S0 N(d1) - K e^{-rT} N(d2)."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S0, K, T, r, sigma):
    """European put: P = K e^{-rT} N(-d2) - S0 N(-d1)."""
    d1, d2 = _d1_d2(S0, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def geometric_asian_call(S0, K, T, r, sigma, n):
    """
    Geometric Asian call, average over t_1, ..., t_n with t_k = kT/n.

    ln G_n is Gaussian with mean mu_G = ln S0 + (r - sigma^2/2) T(n+1)/(2n)
    and variance sigma_G^2 = sigma^2 T(n+1)(2n+1)/(6n^2), hence a Black-Scholes-type formula.
    """
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    s = sigma * np.sqrt(T * (n + 1) * (2 * n + 1) / (6 * n**2))
    d2 = (mu - np.log(K)) / s
    d1 = d2 + s
    return np.exp(-r * T) * (np.exp(mu + 0.5 * s**2) * norm.cdf(d1) - K * norm.cdf(d2))


def up_and_out_call(S0, K, T, r, sigma, B):
    """
    Up-and-out call with continuous monitoring, B > K (Reiner-Rubinstein).

    Computed by in-out parity: C_uo = C_BS - C_ui.
    """
    st = sigma * np.sqrt(T)
    lam = (r + 0.5 * sigma**2) / sigma**2
    x1 = np.log(S0 / B) / st + lam * st
    y = np.log(B**2 / (S0 * K)) / st + lam * st
    y1 = np.log(B / S0) / st + lam * st
    c_ui = (S0 * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - st)
            - S0 * (B / S0)**(2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
            + K * np.exp(-r * T) * (B / S0)**(2 * lam - 2) * (norm.cdf(-y + st) - norm.cdf(-y1 + st)))
    return bs_call_price(S0, K, T, r, sigma) - c_ui
