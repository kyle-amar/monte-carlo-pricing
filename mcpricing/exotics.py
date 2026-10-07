"""Path-dependent options under Black-Scholes: arithmetic Asian and up-and-out barrier calls."""

import numpy as np
from scipy.stats import norm, qmc

from .engine import summarize
from .simulation import brownian_bridge, paths_from_normals, paths_from_W


def asian_payoffs(S, K, T, r):
    """
    Discounted arithmetic (Y) and geometric (X) Asian call payoffs from paths S (N x n).

    X has a closed-form expectation (geometric_asian_call) and serves as a control variate for Y.
    """
    disc = np.exp(-r * T)
    A = S.mean(axis=1)
    G = np.exp(np.log(S).mean(axis=1))
    return disc * np.maximum(A - K, 0.0), disc * np.maximum(G - K, 0.0)


def rqmc_asian(S0, K, T, r, sigma, n, m, n_replications, use_bridge, seed=None):
    """
    Randomized QMC (Sobol + random shift) for the arithmetic Asian call, in dimension n.

    use_bridge=False: incremental construction (coordinate k drives the k-th increment).
    use_bridge=True: Brownian bridge construction (first coordinates drive the large-scale moves).
    """
    rng = np.random.default_rng(seed)
    sobol = qmc.Sobol(d=n, scramble=False).random_base2(m)       # 2^m points in dimension n
    estimates = np.empty(n_replications)
    for j in range(n_replications):
        u = (sobol + rng.uniform(size=n)) % 1.0                  # one shift per coordinate
        Z = norm.ppf(u)
        if use_bridge:
            S = paths_from_W(S0, T, r, sigma, brownian_bridge(Z, T))
        else:
            S = paths_from_normals(S0, T, r, sigma, Z)
        estimates[j] = asian_payoffs(S, K, T, r)[0].mean()
    return summarize(estimates)


def barrier_mc(S0, K, T, r, sigma, B, n_steps, n_paths, method="naive", seed=None):
    """
    Up-and-out call by Monte Carlo, with exact paths at the dates t_k = kT/n_steps.

    method:
      "naive"  - the path is knocked out if some S_{t_k} >= B (biased upwards);
      "bgk"    - Broadie-Glasserman-Kou: barrier shifted to B exp(-0.5826 sigma sqrt(dt));
      "bridge" - payoff weighted by the Brownian-bridge survival probability between dates
                 (unbiased for the continuous barrier, whatever dt).
    """
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    dt = T / n_steps
    B_sim = B * np.exp(-0.5826 * sigma * np.sqrt(dt)) if method == "bgk" else B
    b = np.log(B_sim)

    x = np.full(n_paths, np.log(S0))               # log-price
    alive = np.ones(n_paths, dtype=bool)
    weight = np.ones(n_paths)
    for _ in range(n_steps):
        x_next = x + (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        alive &= (x_next < b)                       # crossing observed at a grid date
        if method == "bridge":                      # crossing between two dates
            p_cross = np.exp(-2 * np.maximum(b - x, 0) * np.maximum(b - x_next, 0) / (sigma**2 * dt))
            weight *= 1 - p_cross
        x = x_next

    payoffs = disc * np.maximum(np.exp(x) - K, 0.0) * alive * weight
    return summarize(payoffs)
