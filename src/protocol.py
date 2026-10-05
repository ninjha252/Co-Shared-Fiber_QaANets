"""Assembling the full per-location protocol channel and the exact logical
fidelity, for arbitrary private operations (Methods: private operations and recovery)."""
import numpy as np

import local_protocol as lp
import optical_channel as oc
import reliability as rl
import steane as st

_GS = None


def recovery_ops():
    global _GS
    if _GS is None:
        _GS = st.build_recovery_operators()[0]
    return _GS


def key_set(kind, n=7, seed=None, axis="y", theta_A=0.0, theta_B=0.0):
    """Return a list of (A_j, B_j) pairs with [A_j, B_j] = 0 for every j."""
    if kind == "identity":
        return [(lp.I2, lp.I2) for _ in range(n)]
    if kind == "fixed_rz":
        return [(lp.rot("z", theta_A), lp.rot("z", theta_B)) for _ in range(n)]
    if kind == "fixed_nonclifford":
        return [(lp.rot(axis, theta_A), lp.rot(axis, theta_B)) for _ in range(n)]
    if kind == "random":
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            ax = "xyz"[int(rng.integers(3))]      # common axis for A and B
            ta, tb = rng.uniform(0, 2 * np.pi, 2)
            out.append((lp.rot(ax, ta), lp.rot(ax, tb)))
        return out
    raise ValueError(kind)


def check_commuting(keys, tol=1e-12):
    return max(float(np.abs(A @ B - B @ A).max()) for A, B in keys)


def location_supers(optical_weights_ixyz, traversal_weights_list, keys):
    return [lp.location_channel_super(optical_weights_ixyz, traversal_weights_list,
                                      A=A, B=B) for A, B in keys]


def location_unitary(A, B, E1, E2, E3):
    """K_j = B^dag E_3 A^dag E_2 B E_1 A  (rightmost acts first)."""
    return B.conj().T @ E3 @ A.conj().T @ E2 @ B @ E1 @ A


def channel_point(eta, nu_pass, e0_pass, z_pass, n=7, t=1):
    """Reduce per-traversal optical and local noise parameters to the scalars of
    the certificate.

    The chain is: compose the three traversals into (T, nu); convert that into
    the accepted optical operation (c0, c1, s, r); convolve its conditional
    Pauli weights with the three traversal channels; read off the
    affected-location probability q and the correctable weight C_{n,t}(q).
    """
    eta = np.asarray(eta, float); nu_pass = np.asarray(nu_pass, float)
    e0_pass = np.atleast_1d(np.asarray(e0_pass, float))
    z_pass = np.atleast_1d(np.asarray(z_pass, float))
    if e0_pass.size == 1:
        e0_pass = np.repeat(e0_pass, 3)
    T, nu, prop = oc.compose_passes(eta, nu_pass)
    res = oc.accepted_optical(T, nu)
    ow = np.array([float(res.wI), float(res.wX), float(res.wY), float(res.wZ)])
    tw = [lp.traversal_weights(e0_pass[m], z_pass[m]) for m in range(3)]
    b = np.array([w[0] for w in tw])
    q = float(lp.affected_probability(res.wI, b))
    C = float(rl.correctable_weight(q, n, t))
    s = float(res.s)
    return {"T": T, "nu": nu, "c0": float(res.c0), "c1": float(res.c1),
            "s": s, "r": float(res.r), "wI": float(res.wI),
            "optical_weights_ixyz": ow, "traversal_weights": tw, "b": b,
            "q": q, "C_bound": C, "P_acc": s ** n, "Y_lb": s ** n * C,
            "Y_ub": s ** n,
            "net_pauli_ixzy": lp.net_pauli_distribution(ow, tw)}


def exact_fidelity(point, keys, method="density_matrix"):
    """Conditional logical entanglement fidelity for the given private ops."""
    if method == "pauli_enumeration":
        if any(not (np.allclose(A, lp.I2) and np.allclose(B, lp.I2)) for A, B in keys):
            raise ValueError("Pauli enumeration requires identity private operations")
        logical = st.pauli_enumeration_logical(point["net_pauli_ixzy"])
        return float(logical[0]), logical
    S = location_supers(point["optical_weights_ixyz"], point["traversal_weights"], keys)
    Fe, rho = st.logical_entanglement_fidelity(S, Gs=recovery_ops())
    return Fe, rho


def uncoded_fidelity(point):
    """n = 1 comparison on the SAME three-pass physical channel.  For a Pauli
    case the exact entanglement fidelity is the net identity probability, not
    1 - q."""
    return float(point["net_pauli_ixzy"][0])


def certificate_threshold_q(F_min, n=7, t=1):
    """Largest affected-location probability q* with C_{n,t}(q*) = F_min."""
    from scipy.optimize import brentq
    return brentq(lambda q: float(rl.correctable_weight(q, n, t)) - F_min,
                  0.0, 0.999999)


def e0_max_for_certification(eta, nu_pass, sigma2, q_star, iters=60):
    """Largest per-traversal base error e0 for which a candidate still meets the
    certificate, for every candidate at once.

    The certificate holds when w_I prod_m b_m(e0) >= 1 - q*, with
    b_m(e0) = (1 - z_m)(1 - 3 e0/2) + z_m e0/2.  The left side decreases
    monotonically in e0 (for z_m < 3/4), so a vectorized bisection on
    e0 in [0, 2/3] is exact to the bisection tolerance.  Candidates that fail
    even at e0 = 0 return NaN: no base-error budget can rescue them.
    """
    T, nu, _ = oc.compose_passes(eta, nu_pass)
    wI = np.asarray(oc.accepted_optical(T, nu).wI, float)
    z = lp.z_from_sigma2(sigma2)

    def lhs(e0):
        b = (1.0 - z) * (1.0 - 1.5 * e0[:, None]) + z * 0.5 * e0[:, None]
        return wI * np.prod(b, axis=1)

    target = 1.0 - q_star
    lo = np.zeros(len(wI)); hi = np.full(len(wI), 2.0 / 3.0)
    feasible = lhs(lo) >= target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = lhs(mid) >= target
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return np.where(feasible, lo, np.nan)
