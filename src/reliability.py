"""Correctable-history weight, acceptance, yield bounds, and their stable
small-number evaluation (Methods: certificate evaluation)."""
import numpy as np
from scipy.stats import binom


def correctable_weight(q, n=7, t=1):
    """C_{n,t}(q) = sum_{l=0}^{t} C(n,l) q^l (1-q)^{n-l}."""
    q = np.asarray(q, dtype=float)
    if np.any(q < -1e-15) or np.any(q > 1 + 1e-15):
        raise ValueError("q outside [0,1]")
    q = np.clip(q, 0.0, 1.0)
    out = np.zeros_like(q)
    from math import comb
    for l in range(t + 1):
        out = out + comb(n, l) * q ** l * (1 - q) ** (n - l)
    return out


def correctable_weight_steane(q):
    """C_{7,1}(q) = (1-q)^7 + 7 q (1-q)^6."""
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    return (1 - q) ** 7 + 7 * q * (1 - q) ** 6


def uncorrectable_weight_steane(q):
    """1 - C_{7,1}(q), computed from the upper tail so that no cancellation of
    two nearly equal numbers occurs at small q."""
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = 6.0 * np.log1p(-q) + np.log1p(6.0 * q)
    return np.where(q >= 1.0, 1.0, -np.expm1(lg))


def correctable_weight_binom(q, n=7, t=1):
    """Independent reference implementation via the binomial CDF."""
    return binom.cdf(t, n, np.asarray(q, dtype=float))


def block_acceptance(s, n=7):
    s = np.asarray(s, dtype=float)
    return s ** n


def log_block_acceptance(s, n=7):
    """log(P_acc) = n log s, retained when the ordinary yield underflows."""
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore"):
        return n * np.log(s)


def yield_bounds(s, q, n=7, t=1):
    """Y_lb = s^n C_{n,t}(q) <= Y <= s^n = Y_ub."""
    s = np.asarray(s, dtype=float)
    C = correctable_weight(q, n, t)
    Pacc = block_acceptance(s, n)
    return Pacc * C, Pacc, C


def markov_correlated_bound(q_marginals, t=1):
    """Pr(|S| <= t) >= max(0, 1 - sum_j q_j/(t+1))  (the correlated-fault control)."""
    return max(0.0, 1.0 - float(np.sum(q_marginals)) / (t + 1))
