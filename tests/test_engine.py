import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student

from conftest import N_SE
from mcpricing import bs_call_price, bs_put_price, call_payoff, mc_price, put_payoff, summarize


@pytest.mark.parametrize("payoff, exact_price", [(call_payoff, bs_call_price), (put_payoff, bs_put_price)])
def test_mc_price_converges_to_black_scholes(params, payoff, exact_price):
    price, se, (lo, hi) = mc_price(payoff, **params, n_paths=1_000_000, seed=42)
    exact = exact_price(**params)
    assert abs(price - exact) < N_SE * se
    assert lo <= exact <= hi
    assert se < 0.02


def test_mc_standard_error_decays_as_one_over_sqrt_n(params):
    _, se_small, _ = mc_price(call_payoff, **params, n_paths=10_000, seed=1)
    _, se_large, _ = mc_price(call_payoff, **params, n_paths=1_000_000, seed=1)
    # 100 times more paths divides the standard error by sqrt(100) = 10
    assert se_small / se_large == pytest.approx(10.0, rel=0.1)


def test_summarize_mean_and_standard_error():
    price, se, _ = summarize([1.0, 2.0, 3.0, 4.0])
    assert price == pytest.approx(2.5)
    assert se == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1) / 2)


def test_summarize_uses_student_quantile_for_small_samples():
    samples = np.random.default_rng(0).standard_normal(16)   # e.g. 16 randomized QMC replications
    price, se, (lo, hi) = summarize(samples)
    q = student.ppf(0.975, df=15)
    assert q == pytest.approx(2.131, abs=1e-3)
    assert (hi - price) / se == pytest.approx(q)
    assert (price - lo) / se == pytest.approx(q)


def test_summarize_quantile_tends_to_gaussian_for_large_samples():
    samples = np.random.default_rng(0).standard_normal(1_000_000)
    price, se, (lo, hi) = summarize(samples)
    assert (hi - price) / se == pytest.approx(norm.ppf(0.975), abs=1e-5)   # 1.95996...


def test_summarize_alpha():
    samples = np.random.default_rng(0).standard_normal(50)
    price, se, (lo, hi) = summarize(samples, alpha=0.01)
    assert (hi - price) / se == pytest.approx(student.ppf(0.995, df=49))
