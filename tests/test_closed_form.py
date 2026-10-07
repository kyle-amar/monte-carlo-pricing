import numpy as np
import pytest

from conftest import N_SE
from mcpricing import bs_call_price, bs_put_price, call_payoff, mc_price, put_payoff


@pytest.mark.parametrize("S0, K, T, r, sigma", [
    (100.0, 100.0, 1.0, 0.03, 0.2),
    (100.0, 70.0, 0.5, 0.01, 0.35),
    (80.0, 120.0, 2.0, 0.05, 0.15),
])
def test_put_call_parity_closed_form(S0, K, T, r, sigma):
    C = bs_call_price(S0, K, T, r, sigma)
    P = bs_put_price(S0, K, T, r, sigma)
    assert C - P == pytest.approx(S0 - K * np.exp(-r * T), abs=1e-10)


def test_put_call_parity_monte_carlo(params):
    c, se_c, _ = mc_price(call_payoff, **params, n_paths=1_000_000, seed=42)
    p, se_p, _ = mc_price(put_payoff, **params, n_paths=1_000_000, seed=42)
    parity = params["S0"] - params["K"] * np.exp(-params["r"] * params["T"])
    # se_c + se_p bounds the standard deviation of C - P whatever the correlation of the two estimators
    assert abs((c - p) - parity) < N_SE * (se_c + se_p)
