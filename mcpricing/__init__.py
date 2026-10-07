"""mcpricing: a Monte Carlo pricing engine for equity derivatives under Black-Scholes."""

from .closed_form import bs_call_price, bs_put_price, geometric_asian_call, up_and_out_call
from .engine import mc_price, summarize
from .exotics import asian_payoffs, barrier_mc, rqmc_asian
from .payoffs import call_payoff, put_payoff
from .simulation import (
    brownian_bridge,
    paths_from_normals,
    paths_from_W,
    simulate_terminal_prices,
    terminal_prices,
    terminal_values_schemes,
)
from .variance_reduction import antithetic_check, mc_antithetic, mc_control_variate, mc_qmc

__version__ = "0.1.0"

__all__ = [
    "call_payoff", "put_payoff",
    "bs_call_price", "bs_put_price", "geometric_asian_call", "up_and_out_call",
    "terminal_prices", "simulate_terminal_prices", "paths_from_normals",
    "brownian_bridge", "paths_from_W", "terminal_values_schemes",
    "summarize", "mc_price",
    "mc_antithetic", "antithetic_check", "mc_control_variate", "mc_qmc",
    "asian_payoffs", "rqmc_asian", "barrier_mc",
]
