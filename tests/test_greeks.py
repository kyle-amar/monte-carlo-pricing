import numpy as np
import pytest

from conftest import N_SE
from mcpricing import (
    bs_call_delta,
    bs_call_gamma,
    bs_call_price,
    bs_call_vega,
    call_payoff,
    digital_delta,
    digital_payoff,
    digital_price,
    fd_delta,
    fd_gamma,
    fd_vega,
    lr_greeks,
    mixed_gamma,
    pathwise_greeks,
)

N = 500_000
EXACT = {"delta": bs_call_delta, "gamma": bs_call_gamma, "vega": bs_call_vega}


@pytest.mark.parametrize("S0, K, T, r, sigma", [
    (100.0, 100.0, 1.0, 0.03, 0.2),
    (100.0, 70.0, 0.5, 0.01, 0.35),
    (80.0, 120.0, 2.0, 0.05, 0.15),
])
def test_closed_form_greeks_match_derivatives_of_closed_form_prices(S0, K, T, r, sigma):
    # The reference Greeks are checked against central differences of the closed-form prices (no Monte Carlo)
    C = lambda s=S0, v=sigma: bs_call_price(s, K, T, r, v)
    h, h_sigma = 1e-3, 1e-5          # sigma ~ 0.2: a smaller step keeps the O(h^2) bias negligible
    assert bs_call_delta(S0, K, T, r, sigma) == pytest.approx((C(S0 + h) - C(S0 - h)) / (2 * h), abs=1e-6)
    assert bs_call_gamma(S0, K, T, r, sigma) == pytest.approx((C(S0 + h) - 2 * C() + C(S0 - h)) / h**2, abs=1e-6)
    assert bs_call_vega(S0, K, T, r, sigma) == pytest.approx(
        (C(v=sigma + h_sigma) - C(v=sigma - h_sigma)) / (2 * h_sigma), abs=1e-6)
    D = lambda s: digital_price(s, K, T, r, sigma)
    assert digital_delta(S0, K, T, r, sigma) == pytest.approx((D(S0 + h) - D(S0 - h)) / (2 * h), abs=1e-6)


@pytest.mark.parametrize("greek", ["delta", "vega"])
def test_pathwise_greeks_match_black_scholes(params, greek):
    est, se, _ = pathwise_greeks(**params, n_paths=N, seed=42)[greek]
    assert abs(est - EXACT[greek](**params)) < N_SE * se


def test_pathwise_gamma_of_call_is_zero(params):
    # The derivative of the indicator 1_{S_T > K} vanishes almost everywhere
    est, se, _ = pathwise_greeks(**params, n_paths=N, seed=42)["gamma"]
    assert est == 0.0
    assert se == 0.0


@pytest.mark.parametrize("greek", ["delta", "gamma", "vega"])
def test_likelihood_ratio_greeks_match_black_scholes(params, greek):
    est, se, _ = lr_greeks(call_payoff, **params, n_paths=N, seed=42)[greek]
    assert abs(est - EXACT[greek](**params)) < N_SE * se


def test_mixed_gamma_matches_black_scholes(params):
    est, se, _ = mixed_gamma(**params, n_paths=N, seed=42)
    assert abs(est - bs_call_gamma(**params)) < N_SE * se


@pytest.mark.parametrize("greek, estimator, h", [
    ("delta", fd_delta, 0.1),
    ("gamma", fd_gamma, 1.0),
    ("vega", fd_vega, 0.01),
])
def test_finite_differences_with_common_random_numbers_match_black_scholes(params, greek, estimator, h):
    # Bias O(h^2): negligible against 4 standard errors for these steps
    est, se, _ = estimator(call_payoff, **params, h=h, n_paths=N, seed=42)
    assert abs(est - EXACT[greek](**params)) < N_SE * se


def test_likelihood_ratio_delta_of_digital_matches_closed_form(params):
    # The payoff is never differentiated: the method works for the discontinuous digital payoff
    est, se, _ = lr_greeks(digital_payoff, **params, n_paths=N, seed=42)["delta"]
    assert abs(est - digital_delta(**params)) < N_SE * se
