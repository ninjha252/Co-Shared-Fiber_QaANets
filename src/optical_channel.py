"""Analytical optical-to-qubit reduction (Methods: optical-to-qubit reduction).

Composition of three thermal attenuators followed by an IDEAL FINAL
single-photon acceptance on the two polarization modes.  No intermediate
photon-number projection is performed.
"""
from dataclasses import dataclass
import numpy as np


def compose_passes(eta, nu):
    """T = eta1 eta2 eta3 ;  nu = eta3 eta2 nu1 + eta3 nu2 + nu3.

    `eta` and `nu` are length-3 sequences ordered by traversal 1,2,3.
    `nu` entries are per-polarization occupancies at each traversal's OUTPUT
    reference plane.  Returns (T, nu_final, propagated_contributions).
    """
    eta = np.asarray(eta, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if eta.shape[-1] != 3 or nu.shape[-1] != 3:
        raise ValueError("expected three traversals")
    T = np.prod(eta, axis=-1)
    w = np.stack([eta[..., 2] * eta[..., 1],
                  eta[..., 2],
                  np.ones_like(eta[..., 2])], axis=-1)
    prop = w * nu
    return T, np.sum(prop, axis=-1), prop


def propagation_weights(eta):
    """Weights (eta2*eta3, eta3, 1) applied to nu_1, nu_2, nu_3."""
    eta = np.asarray(eta, dtype=float)
    return np.stack([eta[..., 2] * eta[..., 1], eta[..., 2],
                     np.ones_like(eta[..., 2])], axis=-1)


@dataclass
class AcceptedOptical:
    T: float
    nu: float
    c0: float
    c1: float
    s: float
    r: float
    wI: float
    wX: float
    wY: float
    wZ: float
    defined: bool          # False when s == 0 (no accepted output)


def accepted_optical(T, nu):
    """Return the accepted optical operation weights for scalar or array input.

    M_opt(rho) = c0 rho + c1 (I/2) Tr rho
    c0 = T/(1+nu)^4 ,  c1 = 2 nu (1+nu-T)/(1+nu)^4
    s  = c0+c1 ,       r  = c1/s
    Conditional Pauli weights (I,X,Y,Z) = (1-3r/4, r/4, r/4, r/4).
    """
    T = np.asarray(T, dtype=float)
    nu = np.asarray(nu, dtype=float)
    if np.any(T < -1e-15) or np.any(T > 1 + 1e-12):
        raise ValueError("T outside [0,1]")
    if np.any(nu < -1e-15):
        raise ValueError("nu must be >= 0")
    T = np.clip(T, 0.0, 1.0)
    nu = np.maximum(nu, 0.0)

    one_plus = 1.0 + nu
    denom4 = one_plus ** 4
    c0 = T / denom4
    c1 = 2.0 * nu * (one_plus - T) / denom4
    # numerically convenient grouping: D = T + 2 nu (1+nu-T), s = D/(1+nu)^4
    D = T + 2.0 * nu * (one_plus - T)
    s = D / denom4
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(D > 0.0, 2.0 * nu * (one_plus - T) / np.where(D > 0, D, 1.0), np.nan)
    defined = D > 0.0
    wI = np.where(defined, 1.0 - 0.75 * r, np.nan)
    wp = np.where(defined, 0.25 * r, np.nan)
    return AcceptedOptical(T=T, nu=nu, c0=c0, c1=c1, s=s, r=r,
                           wI=wI, wX=wp, wY=wp, wZ=wp, defined=defined)


def check_accepted_optical(res, tol=1e-12):
    """Assert the physical domains stated in Section 8.  Returns a dict of residuals."""
    T, nu, s, r = res.T, res.nu, res.s, res.r
    out = {}
    out["T_in_range"] = bool(np.all(T >= -tol) and np.all(T <= 1 + tol))
    out["nu_nonneg"] = bool(np.all(nu >= -tol))
    out["s_in_range"] = bool(np.all(s >= -tol) and np.all(s <= 1 + tol))
    d = res.defined
    if np.any(d):
        rr = np.asarray(r)[np.asarray(d)] if np.ndim(r) else np.array([r])
        out["r_in_range"] = bool(np.all(rr >= -tol) and np.all(rr <= 1 + tol))
        wsum = np.asarray(res.wI) + 3 * np.asarray(res.wX)
        out["weights_normalized_maxdev"] = float(np.nanmax(np.abs(wsum - 1.0)))
        out["weights_nonneg"] = bool(np.nanmin(np.asarray(res.wI)) >= -tol
                                     and np.nanmin(np.asarray(res.wX)) >= -tol)
    return out


def choi_from_weights(c0, c1):
    """Accepted-operation Choi operator with the normalized-reference convention

        J = (id (x) M)(|Phi><Phi|),  |Phi> = (|00>+|11>)/sqrt(2),  Tr J = s.

    Basis order |00>,|01>,|10>,|11> (reference qubit first).
    """
    J = np.zeros((4, 4), dtype=complex)
    # (id (x) M)(|Phi><Phi|) with M(rho)=c0 rho + c1 I/2 Tr rho
    # |Phi><Phi| = 1/2 sum_{u,v} |u u><v v|
    for u in range(2):
        for v in range(2):
            rho_in = np.zeros((2, 2), dtype=complex)
            rho_in[u, v] = 1.0
            out = c0 * rho_in + c1 * 0.5 * np.eye(2) * np.trace(rho_in)
            for m in range(2):
                for n in range(2):
                    J[2 * u + m, 2 * v + n] += 0.5 * out[m, n]
    return J
