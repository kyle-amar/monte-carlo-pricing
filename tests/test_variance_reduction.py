import numpy as np
import pytest

from conftest import N_SE
from mcpricing import (
    antithetic_check,
    bs_call_price,
    bs_put_price,
    call_payoff,
    mc_antithetic,
    mc_control_variate,
    mc_qmc,
    put_payoff,
)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("payoff", [call_payoff, put_payoff], ids=["call", "put"])
def test_antithetic_correlation_is_nonpositive_for_monotone_payoffs(payoff, K):
    rho, factor = antithetic_check(lambda ST: payoff(ST, K))
    assert rho <= 0
    assert factor >= 1   # 1 / (1 + rho): never worse than standard Monte Carlo


def test_antithetic_correlation_is_one_for_even_payoff():
    # h(S_T) = (ln(S_T/S0) - drift)^2 = sigma^2 T Z^2 is even in Z: rho = 1, variance doubled
    S0, T, r, sigma = 100.0, 1.0, 0.03, 0.2
    drift = (r - 0.5 * sigma**2) * T
    rho, factor = antithetic_check(lambda ST: (np.log(ST / S0) - drift)**2)
    assert rho == pytest.approx(1.0)
    assert factor == pytest.approx(0.5)


@pytest.mark.parametrize("payoff, exact_price", [(call_payoff, bs_call_price), (put_payoff, bs_put_price)])
@pytest.mark.parametrize("estimator", [
    lambda payoff, p: mc_antithetic(payoff, **p, n_pairs=2**17, seed=42),
    lambda payoff, p: mc_control_variate(payoff, **p, n_paths=2**18, seed=42),
    lambda payoff, p: mc_control_variate(payoff, **p, n_paths=2**18, seed=42, n_pilot=10_000),
    lambda payoff, p: mc_qmc(payoff, **p, m=14, n_replications=16, seed=42),
], ids=["antithetic", "control_variate", "control_variate_pilot", "rqmc"])
def test_variance_reduction_estimators_match_black_scholes(params, estimator, payoff, exact_price):
    price, se, _ = estimator(payoff, params)
    assert abs(price - exact_price(**params)) < N_SE * se
