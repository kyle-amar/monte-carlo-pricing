"""
Regenerate the README figures (English labels) in figures/ with the mcpricing package.

Same parameters and seeds as the notebooks, so the figures match the README tables.
Usage, from the repository root:  python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

from mcpricing import (
    barrier_mc,
    call_payoff,
    mc_antithetic,
    mc_control_variate,
    mc_price,
    mc_qmc,
    put_payoff,
    up_and_out_call,
)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
PARAMS = dict(S0=100.0, K=100.0, T=1.0, r=0.03, sigma=0.2)
METHODS = ["Standard", "Antithetic", "Control variate", "Randomized QMC"]


def save(name):
    path = FIGURES / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path.relative_to(FIGURES.parent)}")


def standard_errors(payoff, params, N, seed, n_reps_qmc=16):
    """Standard error of each method for a budget of N payoff evaluations (step 2 notebook)."""
    m_qmc = int(np.log2(N // n_reps_qmc))
    return {
        "Standard": mc_price(payoff, **params, n_paths=N, seed=seed)[1],
        "Antithetic": mc_antithetic(payoff, **params, n_pairs=N // 2, seed=seed)[1],
        "Control variate": mc_control_variate(payoff, **params, n_paths=N, seed=seed)[1],
        "Randomized QMC": mc_qmc(payoff, **params, m=m_qmc, n_replications=n_reps_qmc, seed=seed)[1],
    }


def mc_convergence_call_put():
    Ns = np.array([10**k for k in range(2, 8)], dtype=float)
    se_call = [mc_price(call_payoff, **PARAMS, n_paths=int(n), seed=42)[1] for n in Ns]
    se_put = [mc_price(put_payoff, **PARAMS, n_paths=int(n), seed=42)[1] for n in Ns]

    plt.figure(figsize=(7, 5))
    plt.loglog(Ns, se_call, "o-", label="Call")
    plt.loglog(Ns, se_put, "s-", label="Put")
    plt.loglog(Ns, se_call[0] * np.sqrt(Ns[0] / Ns), "k--", label=r"Slope $-1/2$")
    plt.xlabel("Number of simulations N")
    plt.ylabel("Standard error")
    plt.title("Monte Carlo convergence (Black-Scholes)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save("mc_convergence_call_put.png")


def variance_reduction_by_strike():
    strikes = [70, 80, 90, 100, 110, 120, 130]
    vr = {name: [] for name in METHODS[1:]}
    for K in strikes:
        se = standard_errors(call_payoff, dict(PARAMS, K=float(K)), N=2**18, seed=42)
        for name in vr:
            vr[name].append(se["Standard"]**2 / se[name]**2)

    plt.figure(figsize=(7, 5))
    for name, marker in zip(vr, ["o", "s", "^"]):
        plt.semilogy(strikes, vr[name], marker + "-", label=name)
    plt.axhline(1, color="k", ls="--", lw=1, label="Standard (reference)")
    plt.xlabel("Strike K (S0 = 100)")
    plt.ylabel("Variance reduction factor (log scale)")
    plt.title("Variance reduction by strike — European call")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save("variance_reduction_by_strike.png")


def convergence_by_method():
    budgets = [2**k for k in range(10, 21)]
    curves = {name: [] for name in METHODS}
    for n in budgets:
        for name, se in standard_errors(call_payoff, PARAMS, N=n, seed=1).items():
            curves[name].append(se)

    b = np.array(budgets, dtype=float)
    plt.figure(figsize=(7, 5))
    for name, marker in zip(curves, ["o", "s", "^", "D"]):
        plt.loglog(b, curves[name], marker + "-", label=name)
    plt.loglog(b, curves["Standard"][0] * np.sqrt(b[0] / b), "k--", lw=1, label=r"Slope $-1/2$")
    plt.loglog(b, curves["Randomized QMC"][0] * (b[0] / b), "k:", lw=1, label=r"Slope $-1$")
    plt.xlabel("Budget N (payoff evaluations)")
    plt.ylabel("Standard error")
    plt.title("Convergence of the methods — ATM call")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save("convergence_by_method.png")


def barrier_monitoring_bias():
    B = 130.0
    exact = up_and_out_call(**PARAMS, B=B)
    steps = [4, 12, 52, 252]
    labels = {"naive": "Naive", "bgk": "BGK", "bridge": "Brownian bridge"}

    plt.figure(figsize=(7, 5))
    for method, marker in zip(labels, ["o", "s", "^"]):
        results = [barrier_mc(**PARAMS, B=B, n_steps=n, n_paths=200_000, method=method, seed=7) for n in steps]
        p = np.array([res[0] for res in results])
        se = np.array([res[1] for res in results])
        plt.errorbar(steps, p, yerr=1.96 * se, fmt=marker + "-", capsize=3, label=labels[method])
    plt.axhline(exact, color="k", ls="--", lw=1, label="Exact (continuous)")
    plt.xscale("log")
    plt.xlabel("Number of monitoring dates n")
    plt.ylabel("Up-and-out call price")
    plt.title("Discrete monitoring bias and corrections")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    save("barrier_monitoring_bias.png")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    mc_convergence_call_put()
    variance_reduction_by_strike()
    convergence_by_method()
    barrier_monitoring_bias()
