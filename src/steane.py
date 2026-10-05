"""Explicit [[7,1,3]] Steane code, fixed CSS Hamming recovery, and two
independent exact logical-fidelity calculations (Methods: encoder and recovery).

Qubit positions are 1..7 left to right in printed bit strings; position 1 is the
MOST significant bit of the 7-bit integer index, matching numpy tensor axis 0.
"""
import itertools
import numpy as np

from local_protocol import I2, PX, PY, PZ, PAULI_IXZY

N_QUBITS = 7
DIM = 2 ** N_QUBITS

# Parity-check matrix; its columns are the seven nonzero three-bit syndromes.
H_MAT = np.array([[0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 1]], dtype=np.int8)


def _row_span(H):
    rows = H.shape[0]
    span = []
    for coeffs in itertools.product([0, 1], repeat=rows):
        v = np.zeros(H.shape[1], dtype=np.int8)
        for c, row in zip(coeffs, H):
            if c:
                v ^= row
        span.append(v)
    return np.array(span, dtype=np.int8)


CODE_SPAN = _row_span(H_MAT)                      # 8 x 7 binary, the set S


def _bits_to_index(v):
    """Bit string (position 1 first / most significant) -> integer 0..127."""
    idx = 0
    for b in v:
        idx = (idx << 1) | int(b)
    return idx


def _index_to_bits(idx):
    return np.array([(idx >> (N_QUBITS - 1 - i)) & 1 for i in range(N_QUBITS)],
                    dtype=np.int8)


def n_qubit_pauli(x_bits, z_bits):
    """Tensor product of single-qubit Paulis specified by (x,z) per position,
    with the convention X=(1,0), Z=(0,1), Y=(1,1) (Y up to a global phase i)."""
    ops = []
    for x, z in zip(x_bits, z_bits):
        if x and z:
            ops.append(PY)
        elif x:
            ops.append(PX)
        elif z:
            ops.append(PZ)
        else:
            ops.append(I2)
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def build_codewords():
    """|0_L>, |1_L>, and the isometry V = (|0_L>, |1_L>) of shape (128, 2)."""
    zeroL = np.zeros(DIM, dtype=complex)
    for v in CODE_SPAN:
        zeroL[_bits_to_index(v)] = 1.0
    zeroL /= np.sqrt(8.0)
    Xall = n_qubit_pauli(np.ones(N_QUBITS, np.int8), np.zeros(N_QUBITS, np.int8))
    oneL = Xall @ zeroL
    V = np.stack([zeroL, oneL], axis=1)
    return zeroL, oneL, V


def build_stabilizers():
    """Six stabilizer generators ordered (Z-row1,Z-row2,Z-row3,X-row1,X-row2,X-row3).

    The Z-type generators measure s_Z = H x ; the X-type generators measure
    s_X = H z .
    """
    stabs = []
    zero = np.zeros(N_QUBITS, dtype=np.int8)
    for row in H_MAT:                 # Z-type: detect X errors
        stabs.append(n_qubit_pauli(zero, row))
    for row in H_MAT:                 # X-type: detect Z errors
        stabs.append(n_qubit_pauli(row, zero))
    return stabs


def syndrome_of(x_bits, z_bits):
    """Six-bit syndrome (s_Z[0:3], s_X[0:3]) of the Pauli (x,z)."""
    sZ = (H_MAT @ np.asarray(x_bits, dtype=np.int8)) % 2
    sX = (H_MAT @ np.asarray(z_bits, dtype=np.int8)) % 2
    return np.concatenate([sZ, sX]).astype(np.int8)


_COL_LOOKUP = {tuple(H_MAT[:, j]): j for j in range(N_QUBITS)}


def css_correction(s6):
    """Fixed CSS Hamming recovery: return (x_corr, z_corr) binary vectors.

    A nonzero s_Z selects the unique column of H equal to it and applies X
    there; a nonzero s_X selects its column and applies Z there.
    """
    s6 = np.asarray(s6, dtype=np.int8)
    xc = np.zeros(N_QUBITS, dtype=np.int8)
    zc = np.zeros(N_QUBITS, dtype=np.int8)
    sZ, sX = tuple(s6[:3]), tuple(s6[3:])
    if any(sZ):
        xc[_COL_LOOKUP[sZ]] = 1
    if any(sX):
        zc[_COL_LOOKUP[sX]] = 1
    return xc, zc


def in_row_span(v):
    """Is the binary vector v in the row span of H (i.e. a stabilizer support)?"""
    return tuple(np.asarray(v, dtype=np.int8) % 2) in _SPAN_SET


_SPAN_SET = {tuple(v) for v in CODE_SPAN}
_ONES = np.ones(N_QUBITS, dtype=np.int8)


def logical_class(x_res, z_res):
    """Classify a ZERO-SYNDROME residual Pauli into a logical class.

    Returns an index in the (I,X,Z,Y) order (= lx + 2*lz).
    Method 'span': X-part in span(H) means logical X exponent 0, else 1
    (because span(H) union (span(H) xor all-ones) is the full even-syndrome set).
    """
    x_res = np.asarray(x_res, dtype=np.int8) % 2
    z_res = np.asarray(z_res, dtype=np.int8) % 2
    lx = 0 if in_row_span(x_res) else 1
    lz = 0 if in_row_span(z_res) else 1
    return lx + 2 * lz


def logical_class_commutation(x_res, z_res):
    """Alternative classification by commutation with the logical operators.

    For a zero-syndrome residual, the logical X exponent equals its
    anticommutation with Z_L = Z^{(x)7}, and the logical Z exponent its
    anticommutation with X_L = X^{(x)7}.
    """
    x_res = np.asarray(x_res, dtype=np.int8) % 2
    z_res = np.asarray(z_res, dtype=np.int8) % 2
    lx = int(np.sum(x_res * _ONES) % 2)     # anticommutes with Z_L
    lz = int(np.sum(z_res * _ONES) % 2)     # anticommutes with X_L
    return lx + 2 * lz


# ------------------------------------------------------------------ recovery map
def build_recovery_operators():
    """Return the 64 maps G_s = V^dagger D_s Pi_s, each of shape (2, 128),
    together with the projectors' total (for the sum_s Pi_s = I check)."""
    _, _, V = build_codewords()
    stabs = build_stabilizers()
    # Diagonalize: build syndrome label of every computational-like basis is not
    # possible directly (stabilizers are not all diagonal), so build projectors
    # from the product formula.
    Gs = []
    Pi_sum = np.zeros((DIM, DIM), dtype=complex)
    eye = np.eye(DIM, dtype=complex)
    for s in range(64):
        bits = [(s >> (5 - a)) & 1 for a in range(6)]
        # Syndrome projector as the product of the six commuting eigenprojectors
        # (I + (-1)^{s_a} S_a)/2.  The generators are not simultaneously
        # diagonal in the computational basis, so the product form is used
        # rather than a basis relabelling.
        Pi = eye.copy()
        for a, Sa in enumerate(stabs):
            Pi = Pi @ (eye + ((-1) ** bits[a]) * Sa) / 2.0
        Pi_sum += Pi
        xc, zc = css_correction(np.array(bits, dtype=np.int8))
        D = n_qubit_pauli(xc, zc)
        Gs.append(V.conj().T @ D @ Pi)
    return Gs, Pi_sum, V


# ---------------------------------------------- exact calculation A: enumeration
_XBIT = np.array([0, 1, 0, 1], dtype=np.int8)   # x bit of (I,X,Z,Y) index
_ZBIT = np.array([0, 0, 1, 1], dtype=np.int8)   # z bit of (I,X,Z,Y) index
_COL_OF_SYND = np.full(8, -1, dtype=np.int64)
for _j in range(N_QUBITS):
    _COL_OF_SYND[int(H_MAT[0, _j]) * 4 + int(H_MAT[1, _j]) * 2 + int(H_MAT[2, _j])] = _j

_TABLE = None


def _build_enumeration_table():
    """Precompute, for all 4**7 net Pauli strings, the residual logical class
    under the fixed CSS recovery.  Returns (combos, classes, audit)."""
    combos = np.array(list(itertools.product(range(4), repeat=N_QUBITS)), dtype=np.int8)
    x = _XBIT[combos]                       # (4**7, 7)
    z = _ZBIT[combos]
    Ht = H_MAT.T.astype(np.int64)
    sZ = (x.astype(np.int64) @ Ht) % 2      # (M,3) syndromes of the X parts
    sX = (z.astype(np.int64) @ Ht) % 2      # (M,3) syndromes of the Z parts
    # Pack each three-bit syndrome into an integer 0..7 so the corrected qubit
    # is a table lookup (_COL_OF_SYND) instead of a per-string search.
    kZ = sZ[:, 0] * 4 + sZ[:, 1] * 2 + sZ[:, 2]
    kX = sX[:, 0] * 4 + sX[:, 1] * 2 + sX[:, 2]
    xc = np.zeros_like(x)
    zc = np.zeros_like(z)
    nz = np.nonzero(kZ > 0)[0]
    xc[nz, _COL_OF_SYND[kZ[nz]]] = 1
    nz = np.nonzero(kX > 0)[0]
    zc[nz, _COL_OF_SYND[kX[nz]]] = 1
    xr = x ^ xc
    zr = z ^ zc
    resid_sZ = (xr.astype(np.int64) @ Ht) % 2
    resid_sX = (zr.astype(np.int64) @ Ht) % 2
    n_bad = int(np.count_nonzero(resid_sZ.any(axis=1) | resid_sX.any(axis=1)))
    span_keys = {tuple(v) for v in CODE_SPAN}
    lx_span = np.array([0 if tuple(v) in span_keys else 1 for v in xr], dtype=np.int64)
    lz_span = np.array([0 if tuple(v) in span_keys else 1 for v in zr], dtype=np.int64)
    classes_span = lx_span + 2 * lz_span
    lx_c = (xr.sum(axis=1) % 2).astype(np.int64)
    lz_c = (zr.sum(axis=1) % 2).astype(np.int64)
    classes_comm = lx_c + 2 * lz_c
    # The two classifications are independent: membership in span(H) versus
    # anticommutation with the logical operators.  Any mismatch would indicate
    # an error in one of them, so the count is carried into the audit record.
    mism = int(np.count_nonzero(classes_span != classes_comm))
    audit = {"residual_nonzero_syndrome": n_bad, "classification_mismatches": mism,
             "n_strings": int(combos.shape[0])}
    return combos, classes_span, audit


def enumeration_table():
    global _TABLE
    if _TABLE is None:
        _TABLE = _build_enumeration_table()
    return _TABLE


def pauli_enumeration_logical(net_weights_ixzy, return_details=False):
    """Exact logical Pauli distribution for an i.i.d. net physical Pauli channel.

    `net_weights_ixzy` is a length-4 probability vector in (I,X,Z,Y) order,
    index = x + 2*z.  All 4**7 = 16384 net Pauli strings are enumerated.
    Entanglement fidelity of the logical Pauli channel is the identity weight.
    """
    w = np.asarray(net_weights_ixzy, dtype=float)
    if abs(w.sum() - 1.0) > 1e-10:
        raise ValueError("net weights must be normalized")
    combos, classes, audit = enumeration_table()
    p = np.prod(w[combos], axis=1)
    logical = np.bincount(classes, weights=p, minlength=4)
    if return_details:
        det = dict(audit)
        det["total_probability"] = float(p.sum())
        return logical, det
    return logical


# ------------------------------------ exact calculation B: density-matrix route
def encoded_max_entangled():
    """|Omega> = (|0>_R|0_L> + |1>_R|1_L>)/sqrt(2) as a length-256 vector,
    with the reference qubit as the most significant tensor factor."""
    z, o, _ = build_codewords()
    psi = np.zeros(2 * DIM, dtype=complex)
    psi[:DIM] = z
    psi[DIM:] = o
    return psi / np.sqrt(2.0)


def apply_local_super(rho, S, site, n_total=8):
    """Apply a single-qubit superoperator S (4x4, row-major vec convention) to
    tensor `site` of an n_total-qubit density matrix rho (shape 2^n x 2^n)."""
    d = 2 ** n_total
    t = rho.reshape([2] * (2 * n_total))
    S4 = S.reshape(2, 2, 2, 2)          # S[m,n,u,v]
    r_ax = site
    c_ax = n_total + site
    t = np.tensordot(S4, t, axes=([2, 3], [r_ax, c_ax]))
    # tensordot puts the two contracted axes first; this permutation restores
    # the original (row..., column...) axis order of the density matrix.
    remaining = [i for i in range(2 * n_total) if i not in (r_ax, c_ax)]
    pos = {r_ax: 0, c_ax: 1}
    for j, ax in enumerate(remaining):
        pos[ax] = 2 + j
    perm = [pos[i] for i in range(2 * n_total)]
    t = np.transpose(t, axes=perm)
    return t.reshape(d, d)


def logical_entanglement_fidelity(location_supers, Gs=None):
    """Exact conditional logical entanglement fidelity via the 8-qubit route.

    `location_supers` is a list of seven 4x4 superoperators (one per physical
    qubit).  Returns (Fe, rho_RL) where rho_RL is the 4x4 recovered state.
    """
    if Gs is None:
        Gs, _, _ = build_recovery_operators()
    psi = encoded_max_entangled()
    rho = np.outer(psi, psi.conj())
    for j, S in enumerate(location_supers):
        rho = apply_local_super(rho, S, site=1 + j, n_total=8)
    # recovery + decoding on the physical block, reference untouched
    rho_t = rho.reshape(2, DIM, 2, DIM)
    out = np.zeros((2, 2, 2, 2), dtype=complex)
    for G in Gs:
        # (I (x) G) rho (I (x) G)^dagger
        tmp = np.einsum('am,rmsn,bn->rasb', G, rho_t, G.conj())
        out += tmp
    rho_RL = out.reshape(4, 4)
    phi = np.zeros(4, dtype=complex)
    phi[0] = phi[3] = 1 / np.sqrt(2)
    Fe = float(np.real(phi.conj() @ rho_RL @ phi))
    return Fe, rho_RL
