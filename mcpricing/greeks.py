"""
Greeks by Monte Carlo under Black-Scholes: finite differences, pathwise derivatives,
likelihood ratio and a mixed pathwise / likelihood ratio estimator of the gamma.

Every estimator returns the format of summarize: (estimate, standard error, confidence interval),
or a dict of such triples.
"""

import numpy as np

from .engine import summarize
from .simulation import paths_from_normals, terminal_prices


# ---------- Finite differences ----------
def fd_delta(payoff, S0, K, T, r, sigma, h, n_paths, seed=None, common=True):
    """
    Delta by central difference (C(S0 + h) - C(S0 - h)) / 2h, bias O(h^2).

    common=True: common random numbers (the same Z for both prices), variance O(1/N);
    common=False: independent draws, variance of order 1/(N h^2).
    """
    rng = np.random.default_rng(seed)
    Z_up = rng.standard_normal(n_paths)
    Z_down = Z_up if common else rng.standard_normal(n_paths)
    disc = np.exp(-r * T)
    up = disc * payoff(terminal_prices(S0 + h, T, r, sigma, Z_up), K)
    down = disc * payoff(terminal_prices(S0 - h, T, r, sigma, Z_down), K)
    return summarize((up - down) / (2 * h))


def fd_gamma(payoff, S0, K, T, r, sigma, h, n_paths, seed=None):
    """Gamma by central second difference (C(S0 + h) - 2 C(S0) + C(S0 - h)) / h^2, with common random numbers."""
    Z = np.random.default_rng(seed).standard_normal(n_paths)
    disc = np.exp(-r * T)
    p = lambda s: disc * payoff(terminal_prices(s, T, r, sigma, Z), K)
    return summarize((p(S0 + h) - 2 * p(S0) + p(S0 - h)) / h**2)


def fd_vega(payoff, S0, K, T, r, sigma, h, n_paths, seed=None):
    """Vega by central difference on sigma, with common random numbers."""
    Z = np.random.default_rng(seed).standard_normal(n_paths)
    disc = np.exp(-r * T)
    up = disc * payoff(terminal_prices(S0, T, r, sigma + h, Z), K)
    down = disc * payoff(terminal_prices(S0, T, r, sigma - h, Z), K)
    return summarize((up - down) / (2 * h))


# ---------- Pathwise derivatives ----------
def pathwise_greeks(S0, K, T, r, sigma, n_paths, seed=None):
    """
    Pathwise Greeks of the European call: the payoff is differentiated inside the expectation.

    delta = E[e^{-rT} 1_{S_T > K} S_T / S0], vega = E[e^{-rT} 1_{S_T > K} S_T (-sigma T + sqrt(T) Z)].
    The pathwise gamma is 0: the derivative of the indicator vanishes almost everywhere.
    """
    Z = np.random.default_rng(seed).standard_normal(n_paths)
    ST = terminal_prices(S0, T, r, sigma, Z)
    itm = (ST > K)
    disc = np.exp(-r * T)
    delta = disc * itm * ST / S0
    vega = disc * itm * ST * (-sigma * T + np.sqrt(T) * Z)
    gamma = np.zeros(n_paths)          # derivative of the indicator: 0 almost everywhere
    return {"delta": summarize(delta), "gamma": summarize(gamma), "vega": summarize(vega)}


# ---------- Likelihood ratio ----------
def lr_greeks(payoff, S0, K, T, r, sigma, n_paths, seed=None):
    """
    Likelihood ratio Greeks: E[e^{-rT} h(S_T) * score], valid for any payoff (even discontinuous).

    Scores, written with Z:  delta  Z / (S0 sigma sqrt(T)),
                             gamma  (Z^2 - 1) / (S0^2 sigma^2 T) - Z / (S0^2 sigma sqrt(T)),
                             vega   (Z^2 - 1) / sigma - Z sqrt(T).
    """
    Z = np.random.default_rng(seed).standard_normal(n_paths)
    ST = terminal_prices(S0, T, r, sigma, Z)
    Y = np.exp(-r * T) * payoff(ST, K)
    st = sigma * np.sqrt(T)
    w_delta = Z / (S0 * st)
    w_gamma = (Z**2 - 1) / (S0**2 * st**2) - Z / (S0**2 * st)
    w_vega = (Z**2 - 1) / sigma - Z * np.sqrt(T)
    return {"delta": summarize(Y * w_delta), "gamma": summarize(Y * w_gamma), "vega": summarize(Y * w_vega)}


def mixed_gamma(S0, K, T, r, sigma, n_paths, seed=None):
    """
    Call gamma, pathwise once then likelihood ratio once:
    E[e^{-rT} 1_{S_T > K} S_T / S0^2 (Z / (sigma sqrt(T)) - 1)].
    """
    Z = np.random.default_rng(seed).standard_normal(n_paths)
    ST = terminal_prices(S0, T, r, sigma, Z)
    g = np.exp(-r * T) * (ST > K) * ST / S0**2 * (Z / (sigma * np.sqrt(T)) - 1)
    return summarize(g)


# ---------- Arithmetic Asian call ----------
def asian_call_delta(S0, K, T, r, sigma, n_dates, n_paths, h, seed=None):
    """
    Delta of the arithmetic Asian call (average over t_k = kT/n_dates) by the three methods,
    on the same Gaussian increments:

      "pathwise": E[e^{-rT} 1_{A_n > K} A_n / S0];
      "lr":       E[e^{-rT} (A_n - K)^+ Z_1 / (S0 sigma sqrt(dt))] (S0 only enters the first step);
      "fd":       central difference with common random numbers and step h.
    """
    disc = np.exp(-r * T)
    dt = T / n_dates
    Zp = np.random.default_rng(seed).standard_normal((n_paths, n_dates))
    A = paths_from_normals(S0, T, r, sigma, Zp).mean(axis=1)

    pw_a = summarize(disc * (A > K) * A / S0)
    lr_a = summarize(disc * np.maximum(A - K, 0) * Zp[:, 0] / (S0 * sigma * np.sqrt(dt)))
    A_up = paths_from_normals(S0 + h, T, r, sigma, Zp).mean(axis=1)
    A_down = paths_from_normals(S0 - h, T, r, sigma, Zp).mean(axis=1)
    fd_a = summarize(disc * (np.maximum(A_up - K, 0) - np.maximum(A_down - K, 0)) / (2 * h))
    return {"pathwise": pw_a, "lr": lr_a, "fd": fd_a}
