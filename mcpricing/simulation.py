"""
Simulation of the Black-Scholes model under the risk-neutral measure Q.

The Gaussian draws are passed as inputs wherever possible, so the same functions
work with pseudo-random, antithetic or (randomized) Sobol normals.
"""

import numpy as np


def terminal_prices(S0, T, r, sigma, Z):
    """Exact S_T under Q from given standard normals Z (pseudo-random, antithetic, Sobol...)."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_terminal_prices(S0, T, r, sigma, n_paths, rng):
    """Exact simulation of S_T under Q (no discretization), drawing n_paths normals from rng."""
    return terminal_prices(S0, T, r, sigma, rng.standard_normal(n_paths))


def paths_from_normals(S0, T, r, sigma, Z):
    """
    Exact Black-Scholes paths at t_1, ..., t_n from Gaussian increments Z (N x n).

    Chains the exact one-step formula: log S_{k+1} = log S_k + (r - sigma^2/2) dt + sigma sqrt(dt) Z_k.
    Returns an N x n array whose column k is S_{t_{k+1}}.
    """
    n_steps = Z.shape[1]
    dt = T / n_steps
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return S0 * np.exp(np.cumsum(log_increments, axis=1))


def brownian_bridge(Z, T):
    """
    Brownian motion W at t_1, ..., t_n (n a power of 2), built by Brownian bridge.

    Z[:, 0] sets W_T, Z[:, 1] sets W_{T/2}, then W_{T/4} and W_{3T/4}, and so on:
    the large-scale moves are carried by the first coordinates (useful for QMC).
    """
    N, n = Z.shape
    assert n & (n - 1) == 0, "n must be a power of 2"
    W = np.zeros((N, n + 1))                       # W[:, k] = W_{t_k}, with W_{t_0} = 0
    W[:, n] = np.sqrt(T) * Z[:, 0]
    j, h = 1, n
    while h > 1:
        half = h // 2
        for left in range(0, n, h):
            mid, right = left + half, left + h
            t_l, t_m, t_r = left * T / n, mid * T / n, right * T / n
            mean = ((t_r - t_m) * W[:, left] + (t_m - t_l) * W[:, right]) / (t_r - t_l)
            var = (t_m - t_l) * (t_r - t_m) / (t_r - t_l)
            W[:, mid] = mean + np.sqrt(var) * Z[:, j]
            j += 1
        h = half
    return W[:, 1:]


def paths_from_W(S0, T, r, sigma, W):
    """S_{t_k} = S0 exp((r - sigma^2/2) t_k + sigma W_{t_k}) on the grid t_k = kT/n."""
    n = W.shape[1]
    t_grid = np.arange(1, n + 1) * T / n
    return S0 * np.exp((r - 0.5 * sigma**2) * t_grid + sigma * W)


def terminal_values_schemes(S0, T, r, sigma, n_paths, n_steps, seed=None):
    """
    S_T computed exactly, by the Euler scheme and by the Milstein scheme, from the SAME Brownian increments.

    Euler:    S_{k+1} = S_k (1 + r dt + sigma dW_k)
    Milstein: S_{k+1} = S_k (1 + r dt + sigma dW_k + sigma^2/2 (dW_k^2 - dt))
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S_exact = np.full(n_paths, S0)
    S_euler = np.full(n_paths, S0)
    S_milstein = np.full(n_paths, S0)
    for _ in range(n_steps):                       # loop over time: O(N) memory only
        dW = np.sqrt(dt) * rng.standard_normal(n_paths)
        S_exact = S_exact * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        S_euler = S_euler * (1 + r * dt + sigma * dW)
        S_milstein = S_milstein * (1 + r * dt + sigma * dW + 0.5 * sigma**2 * (dW**2 - dt))
    return S_exact, S_euler, S_milstein
