"""Vanilla and digital payoffs, written as functions of the terminal price S_T."""

import numpy as np


def call_payoff(ST, K):
    """Call payoff (S_T - K)^+."""
    return np.maximum(ST - K, 0.0)


def put_payoff(ST, K):
    """Put payoff (K - S_T)^+."""
    return np.maximum(K - ST, 0.0)


def digital_payoff(ST, K):
    """Cash-or-nothing digital call payoff 1_{S_T > K}."""
    return (ST > K).astype(float)
