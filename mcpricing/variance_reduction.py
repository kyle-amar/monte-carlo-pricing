"""
Variance reduction for European payoffs under Black-Scholes.

Every estimator returns the same format as summarize: (price, standard error, confidence interval).
"""

import numpy as np
from scipy.stats import norm, qmc

from .engine import summarize
from .simulation import terminal_prices


def mc_antithetic(payoff, S0, K, T, r, sigma, n_pairs, seed=None):
    """
    Antithetic variates: each Z is paired with -Z.

    The standard error is computed on the n_pairs pair means (independent), not on the
    2 n_pairs payoffs (pairwise correlated). Cost: 2 n_pairs payoff evaluations.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_pairs)
    disc = np.exp(-r * T)
    Y_plus = disc * payoff(terminal_prices(S0, T, r, sigma, Z), K)
    Y_minus = disc * payoff(terminal_prices(S0, T, r, sigma, -Z), K)
    pair_means = 0.5 * (Y_plus + Y_minus)      # n_pairs independent samples
    return summarize(pair_means)


def antithetic_check(payoff_of_ST, S0=100.0, T=1.0, r=0.03, sigma=0.2, n=1_000_000, seed=0):
    """
    Correlation rho between g(Z) and g(-Z), and theoretical variance reduction factor 1/(1 + rho).

    payoff_of_ST is a function of S_T only. rho <= 0 is guaranteed when the payoff is monotone.
    """
    Z = np.random.default_rng(seed).standard_normal(n)
    disc = np.exp(-r * T)
    Y_plus = disc * payoff_of_ST(terminal_prices(S0, T, r, sigma, Z))
    Y_minus = disc * payoff_of_ST(terminal_prices(S0, T, r, sigma, -Z))
    rho = np.corrcoef(Y_plus, Y_minus)[0, 1]
    return rho, 1 / (1 + rho)


def _estimate_b(Y, X):
    """Optimal coefficient b* = Cov(Y, X) / Var(X) (slope of the regression of Y on X)."""
    cov = np.cov(Y, X, ddof=1)
    return cov[0, 1] / cov[1, 1]


def mc_control_variate(payoff, S0, K, T, r, sigma, n_paths, seed=None, n_pilot=0):
    """
    Control variate X = e^{-rT} S_T, whose expectation S0 is known (martingale property).

    n_pilot = 0: b* estimated on the same paths (O(1/N) bias, negligible in practice).
    n_pilot > 0: b* estimated on an independent pilot run (exactly unbiased estimator).
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    if n_pilot > 0:   # b* estimated on independent paths
        ST_pilot = terminal_prices(S0, T, r, sigma, rng.standard_normal(n_pilot))
        b = _estimate_b(disc * payoff(ST_pilot, K), disc * ST_pilot)

    ST = terminal_prices(S0, T, r, sigma, rng.standard_normal(n_paths))
    Y = disc * payoff(ST, K)
    X = disc * ST                     # E[X] = S0 exactly

    if n_pilot == 0:  # b* estimated on the same paths
        b = _estimate_b(Y, X)

    Y_cv = Y - b * (X - S0)
    return summarize(Y_cv)


def mc_qmc(payoff, S0, K, T, r, sigma, m, n_replications=16, seed=None):
    """
    Randomized QMC: n_replications independent estimates, each on 2^m Sobol points
    shifted by a uniform random shift modulo 1 (Cranley-Patterson rotation).

    The standard error is computed on the n_replications estimates.
    """
    rng = np.random.default_rng(seed)
    sobol = qmc.Sobol(d=1, scramble=False).random_base2(m)[:, 0]   # 2^m deterministic points
    disc = np.exp(-r * T)

    estimates = np.empty(n_replications)
    for j in range(n_replications):
        u = (sobol + rng.uniform()) % 1.0          # random shift
        Z = norm.ppf(u)
        estimates[j] = (disc * payoff(terminal_prices(S0, T, r, sigma, Z), K)).mean()
    return summarize(estimates)
