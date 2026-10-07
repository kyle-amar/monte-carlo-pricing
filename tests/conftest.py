import pytest

# Number of standard errors allowed between a Monte Carlo estimate and its exact value.
# Seeds are fixed, so every test is deterministic; with any other seed, a correct
# estimator would still fail a 4-sigma check with probability about 6e-5 only.
N_SE = 4


@pytest.fixture
def params():
    """Reference Black-Scholes parameters used throughout the notebooks."""
    return dict(S0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2)
