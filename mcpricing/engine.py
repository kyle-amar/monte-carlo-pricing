"""Generic Monte Carlo engine and the statistics shared by every estimator."""

import numpy as np
from scipy.stats import t as student

from .simulation import simulate_terminal_prices


def summarize(samples, alpha=0.05):
    """
    Price, standard error and (1 - alpha) confidence interval from i.i.d. samples
    whose expectation is the target price.

    Uses the Student quantile with n - 1 degrees of freedom: essential for randomized QMC
    (n = 16 replications), and equal to the Gaussian quantile 1.96 for large samples.
    """
    samples = np.asarray(samples)
    n = len(samples)
    price = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(n)                # sigma_Y_hat / sqrt(n)
    q = student.ppf(1 - alpha / 2, df=n - 1)
    return price, se, (price - q * se, price + q * se)


def mc_price(payoff, S0, K, T, r, sigma, n_paths, seed=None, alpha=0.05):
    """
    Monte Carlo estimator of e^{-rT} E[payoff(S_T, K)] under Black-Scholes.

    Returns (estimated price, standard error, (1 - alpha) confidence interval).
    """
    rng = np.random.default_rng(seed)
    ST = simulate_terminal_prices(S0, T, r, sigma, n_paths, rng)
    discounted_payoffs = np.exp(-r * T) * payoff(ST, K)
    return summarize(discounted_payoffs, alpha)
