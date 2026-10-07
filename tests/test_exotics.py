import numpy as np
import pytest

from conftest import N_SE
from mcpricing import (
    asian_payoffs,
    barrier_mc,
    brownian_bridge,
    geometric_asian_call,
    paths_from_normals,
    summarize,
    up_and_out_call,
)


@pytest.mark.parametrize("n", [12, 52])
def test_geometric_asian_closed_form_matches_monte_carlo(params, n):
    S0, K, T, r, sigma = params.values()
    Z = np.random.default_rng(10 + n).standard_normal((200_000, n))
    _, X = asian_payoffs(paths_from_normals(S0, T, r, sigma, Z), K, T, r)
    price, se, _ = summarize(X)
    assert abs(price - geometric_asian_call(S0, K, T, r, sigma, n)) < N_SE * se


@pytest.mark.parametrize("n_steps", [4, 52])
def test_barrier_brownian_bridge_matches_closed_form(params, n_steps):
    # The Brownian-bridge estimator is unbiased for continuous monitoring whatever the time step
    price, se, _ = barrier_mc(**params, B=130.0, n_steps=n_steps, n_paths=200_000, method="bridge", seed=7)
    assert abs(price - up_and_out_call(**params, B=130.0)) < N_SE * se


def test_barrier_naive_estimator_overestimates_continuous_price(params):
    # Discrete monitoring misses crossings between dates: the naive price is biased upwards
    price, se, _ = barrier_mc(**params, B=130.0, n_steps=4, n_paths=200_000, method="naive", seed=7)
    assert price - up_and_out_call(**params, B=130.0) > N_SE * se


def test_brownian_bridge_has_brownian_motion_covariance():
    T = 1.0
    W = brownian_bridge(np.random.default_rng(0).standard_normal((200_000, 16)), T)
    assert W[:, 3].var() == pytest.approx(T / 4, rel=0.02)                              # Var(W_{T/4}) = T/4
    assert np.corrcoef(W[:, 7], W[:, -1])[0, 1] == pytest.approx(np.sqrt(0.5), abs=0.01)  # Corr(W_{T/2}, W_T)
