"""Independent optical validation by thermal-environment beam-splitter dilation
(Methods: independent optical construction).

Nothing here calls the closed-form c0/c1 expressions.  Each polarization mode is
mixed on a beam splitter of power transmission T with its own independent
thermal environment of mean occupation nbar = nu/(1-T); the environments are
traced out and the two signal polarizations are projected onto total photon
number one.

The beam-splitter unitary is built exactly inside each total-photon-number
sector by exponentiating the finite antisymmetric generator of that sector, so
no ladder operator is truncated inside a retained sector.
"""
import numpy as np
from scipy.linalg import expm

# qubit label -> (H occupation, V occupation)
_HW = {0: (1, 0), 1: (0, 1)}


def bs_sector(N, theta):
    """<m, N-m| U |u, N-u> for U = exp(theta (a^dag e - a e^dag)).

    Returns the (N+1)x(N+1) real matrix indexed by the first-mode occupation.
    Under this convention the Heisenberg output mode is
    a_out = a cos(theta) + e sin(theta), i.e. signal transmission cos^2(theta).
    """
    G = np.zeros((N + 1, N + 1), dtype=float)
    for n in range(N + 1):
        m = N - n
        if m >= 1:                       # a^dag e : |n,m> -> |n+1,m-1>
            G[n + 1, n] += np.sqrt((n + 1) * m)
        if n >= 1:                       # -a e^dag : |n,m> -> |n-1,m+1>
            G[n - 1, n] -= np.sqrt(n * (m + 1))
    return expm(theta * G)


def thermal_weights(nbar, tail_tol=1e-16, m_max=400):
    """Truncated thermal photon-number distribution and its discarded tail.

    delta_th = (nbar/(1+nbar))**(M+1) is the exact discarded probability.
    """
    if nbar < 0:
        raise ValueError("nbar must be nonnegative")
    if nbar == 0.0:
        return np.array([1.0]), 0.0, 0
    ratio = nbar / (1.0 + nbar)
    M = int(np.ceil(np.log(tail_tol) / np.log(ratio))) - 1
    M = max(0, min(M, m_max))
    j = np.arange(M + 1)
    p = ratio ** j / (1.0 + nbar)
    delta = ratio ** (M + 1)
    return p, float(delta), M


def single_mode_elements(T, nu, dim=2, tail_tol=1e-16):
    """Matrix elements Ech[a, d, b, c] = <a| E(|b><c|) |d> of one polarization's
    thermal attenuator, for a,b,c,d in range(dim).

    Built entirely from the beam-splitter dilation with a truncated thermal
    environment; no closed-form channel expression is used.
    """
    if not (0.0 <= T < 1.0):
        raise ValueError("dilation requires 0 <= T < 1 (T=1 is a separate limit)")
    nbar = nu / (1.0 - T) if nu > 0 else 0.0
    p, delta, M = thermal_weights(nbar, tail_tol)
    theta = np.arccos(np.sqrt(T))
    n_max = dim - 1 + M
    BS = [bs_sector(N, theta) for N in range(n_max + 1)]

    Ech = np.zeros((dim, dim, dim, dim), dtype=complex)
    for b in range(dim):
        for c in range(dim):
            for a in range(dim):
                d = c - b + a                     # required by photon counting
                if d < 0 or d >= dim:
                    continue
                acc = 0.0
                for j, pj in enumerate(p):
                    k = b + j - a
                    if k < 0:
                        continue
                    N1, N2 = b + j, c + j
                    acc += pj * BS[N1][a, b] * BS[N2][d, c]
                Ech[a, d, b, c] = acc
    return Ech, {"nbar": nbar, "env_cutoff_M": M, "discarded_tail": delta,
                 "two_env_tail_bound": 2 * delta, "theta": theta}


def accepted_map_numeric(T, nu, rho, Ech=None, tail_tol=1e-16):
    """Unnormalized accepted qubit output for an input qubit density matrix rho.

    Both polarizations use independent copies of the same single-mode channel;
    the output is projected onto total photon number one.
    """
    if Ech is None:
        Ech, _ = single_mode_elements(T, nu, dim=2, tail_tol=tail_tol)
    out = np.zeros((2, 2), dtype=complex)
    for m in range(2):
        hm, wm = _HW[m]
        for n in range(2):
            hn, wn = _HW[n]
            acc = 0.0 + 0j
            for u in range(2):
                hu, wu = _HW[u]
                for v in range(2):
                    hv, wv = _HW[v]
                    acc += rho[u, v] * Ech[hm, hn, hu, hv] * Ech[wm, wn, wu, wv]
            out[m, n] = acc
    return out


def accepted_choi_numeric(T, nu, tail_tol=1e-16):
    """Accepted-operation Choi operator J = (id (x) M)(|Phi><Phi|), Tr J = s,
    built from the four qubit basis operators so off-diagonal information is kept.
    Basis order |00>,|01>,|10>,|11> with the reference qubit first."""
    Ech, info = single_mode_elements(T, nu, dim=2, tail_tol=tail_tol)
    J = np.zeros((4, 4), dtype=complex)
    for u in range(2):
        for v in range(2):
            rho_in = np.zeros((2, 2), dtype=complex)
            rho_in[u, v] = 1.0
            out = accepted_map_numeric(T, nu, rho_in, Ech=Ech)
            J[2 * u:2 * u + 2, 2 * v:2 * v + 2] += 0.5 * out
    return J, Ech, info


# ------------------------------------------------------------------ composition
def fock_channel_super(T, nu, dim, tail_tol=1e-16):
    """Full truncated-Fock superoperator of one polarization's thermal
    attenuator: E(rho)[a,d] = sum_{b,c} Ech[a,d,b,c] rho[b,c]."""
    Ech, info = single_mode_elements(T, nu, dim=dim, tail_tol=tail_tol)
    S = Ech.transpose(0, 1, 2, 3).reshape(dim * dim, dim * dim)
    return S, info


def three_traversal_accepted(eta, nu_pass, dim=6, tail_tol=1e-16):
    """Propagate through three explicit optical traversals on the Fock space,
    keeping every intermediate multiphoton sector, and project only at the end.

    Returns the unnormalized accepted qubit Choi operator.
    """
    supers = []
    infos = []
    for T_m, nu_m in zip(eta, nu_pass):
        S, info = fock_channel_super(T_m, nu_m, dim=dim, tail_tol=tail_tol)
        supers.append(S)
        infos.append(info)

    def apply_chain(rho_full):
        v = rho_full.reshape(-1)
        for S in supers:
            v = S @ v
        return v.reshape(dim, dim)

    # cache the three-pass action on the four single-mode basis operators
    basis_out = {}
    for b in range(2):
        for c in range(2):
            rho0 = np.zeros((dim, dim), dtype=complex)
            rho0[b, c] = 1.0
            basis_out[(b, c)] = apply_chain(rho0)

    J = np.zeros((4, 4), dtype=complex)
    for u in range(2):
        hu, wu = _HW[u]
        for v in range(2):
            hv, wv = _HW[v]
            outH = basis_out[(hu, hv)]
            outV = basis_out[(wu, wv)]
            blk = np.zeros((2, 2), dtype=complex)
            for m in range(2):
                hm, wm = _HW[m]
                for n in range(2):
                    hn, wn = _HW[n]
                    blk[m, n] = outH[hm, hn] * outV[wm, wn]
            J[2 * u:2 * u + 2, 2 * v:2 * v + 2] += 0.5 * blk
    return J, infos
