"""Base imperfections, differential XPM dephasing, private operations, and the
affected-location parameter q  (Methods: private operations and traversal noise).

Pauli index conventions
-----------------------
`PAULI_IXYZ`  : manuscript printed order (I,X,Y,Z) -> indices 0,1,2,3.
`PAULI_IXZY`  : XOR-convolution order  (I,X,Z,Y) -> integers 0,1,2,3, where the
                integer is the two-bit code (x,z) with x the low bit.
Both orders are used; conversion helpers are provided and tested against each
other.
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_IXYZ = [I2, PX, PY, PZ]
# XOR/(x,z) order: 0=(0,0)=I, 1=(1,0)=X, 2=(0,1)=Z, 3=(1,1)=Y
PAULI_IXZY = [I2, PX, PZ, PY]
IXYZ_TO_IXZY = np.array([0, 1, 3, 2])   # (I,X,Y,Z) position -> (I,X,Z,Y) position
IXZY_TO_IXYZ = np.array([0, 1, 3, 2])   # involution


def ixyz_to_ixzy(w):
    """Reorder a length-4 weight vector from (I,X,Y,Z) to (I,X,Z,Y)."""
    w = np.asarray(w, dtype=float)
    out = np.empty_like(w)
    out[IXYZ_TO_IXZY] = w
    return out


def ixzy_to_ixyz(w):
    w = np.asarray(w, dtype=float)
    out = np.empty_like(w)
    out[IXZY_TO_IXYZ] = w
    return out


def z_from_sigma2(sigma2):
    """Z-flip probability from a zero-mean Gaussian differential phase variance:
        z = (1 - exp(-sigma^2/2))/2 .
    """
    sigma2 = np.asarray(sigma2, dtype=float)
    if np.any(sigma2 < 0):
        raise ValueError("sigma^2 must be nonnegative")
    return -0.5 * np.expm1(-0.5 * sigma2)


def sigma2_from_z(z):
    """Inverse of `z_from_sigma2`; requires z < 1/2."""
    z = np.asarray(z, dtype=float)
    if np.any(z >= 0.5) or np.any(z < 0):
        raise ValueError("z must lie in [0, 1/2)")
    return -2.0 * np.log1p(-2.0 * z)


def traversal_weights(e0, z):
    """Composed base + XPM traversal Pauli weights in (I,X,Y,Z) order.

    a0 = e0/2 with 0 <= a0 <= 1/3.  Base channel (1-3a0, a0, a0, a0)
    composed with a Z flip of probability z.
    """
    e0, z = np.broadcast_arrays(np.asarray(e0, dtype=float),
                                np.asarray(z, dtype=float))
    a0 = 0.5 * e0
    if np.any(a0 < 0) or np.any(a0 > 1.0 / 3.0 + 1e-15):
        raise ValueError("a0 = e0/2 must lie in [0, 1/3]")
    if np.any(z < 0) or np.any(z > 1):
        raise ValueError("z must lie in [0,1]")
    pI = (1 - z) * (1 - 3 * a0) + z * a0
    pX = a0
    pY = a0
    pZ = z * (1 - 3 * a0) + (1 - z) * a0
    return np.stack([pI, pX, pY, pZ], axis=-1)


def affected_probability(wI, b):
    """Affected-location probability q = 1 - w_I prod_m b_m."""
    wI = np.asarray(wI, dtype=float)
    b = np.asarray(b, dtype=float)
    return 1.0 - wI * np.prod(b, axis=-1)


# ---------------------------------------------------------------- superoperators
def pauli_channel_super(weights_ixyz):
    """4x4 superoperator S with E(rho)_{mn} = sum_{uv} S[(m,n),(u,v)] rho_{uv}.

    Row index is 2*m+n, column index is 2*u+v (row-major vec).
    """
    w = np.asarray(weights_ixyz, dtype=float)
    S = np.zeros((4, 4), dtype=complex)
    for p, P in zip(w, PAULI_IXYZ):
        S += p * np.kron(P, P.conj())
    return S


def unitary_super(U):
    """Superoperator of rho -> U rho U^dagger in the same row-major convention."""
    return np.kron(U, U.conj())


def apply_super(S, rho):
    return (S @ rho.reshape(4)).reshape(2, 2)


def rot(axis, theta):
    """R_axis(theta) = exp(-i theta P_axis / 2)."""
    P = {"x": PX, "y": PY, "z": PZ}[axis.lower()]
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * P


def location_channel_super(optical_weights_ixyz, traversal_weights_list,
                           A=None, B=None):
    """Superoperator of

        N_j = Lambda_opt . U_{B^dag} . B_3 . U_{A^dag} . B_2 . U_B . B_1 . U_A

    (rightmost acts first).  `traversal_weights_list` holds the three composed
    base+XPM traversal weight vectors in (I,X,Y,Z) order.
    """
    if A is None:
        A = I2
    if B is None:
        B = I2
    B1, B2, B3 = [pauli_channel_super(w) for w in traversal_weights_list]
    Lo = pauli_channel_super(optical_weights_ixyz)
    seq = [unitary_super(A), B1, unitary_super(B), B2,
           unitary_super(A.conj().T), B3, unitary_super(B.conj().T), Lo]
    S = np.eye(4, dtype=complex)
    for M in seq:                      # left-multiply in application order
        S = M @ S
    return S


def net_pauli_distribution(optical_weights_ixyz, traversal_weights_list):
    """Net single-location Pauli distribution for IDENTITY private operations.

    Convolves the optical channel with the three traversal channels using
    Pauli multiplication modulo global phase.  Returns weights in the
    (I,X,Z,Y) integer order (index = x + 2z).
    """
    dists = [ixyz_to_ixzy(optical_weights_ixyz)] + \
            [ixyz_to_ixzy(w) for w in traversal_weights_list]
    out = np.zeros(4)
    out[0] = 1.0
    for d in dists:
        new = np.zeros(4)
        for i in range(4):
            if out[i] == 0.0:
                continue
            for j in range(4):
                new[i ^ j] += out[i] * d[j]
        out = new
    return out
