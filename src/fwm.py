"""Four-wave mixing products from the frozen classical grid (Methods: four-wave mixing).

Only FORWARD-generated products are modeled, and they are received only at the
terminal the classical comb propagates towards.  Backward generation,
reflection, and Rayleigh backscatter are NOT modeled; a return-pass receiver
therefore collects no modeled FWM contribution.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class FWMProducts:
    freq_hz: np.ndarray          # unique generated frequencies
    power_incoherent_w: np.ndarray
    power_coherent_w: np.ndarray
    n_triplets: int
    n_lines: int
    audit: dict


def _long_integral_closed(a_eff, dbeta, L, a_out):
    """int_0^L exp(-a_eff z) exp(i dbeta z) exp(-a_out (L-z)/2) dz
    for constant coefficients, evaluated stably."""
    x = (a_eff - 0.5 * a_out) - 1j * dbeta
    small = np.abs(x * L) < 1e-10
    safe = np.where(small, 1.0, x)
    core = np.where(small, L, (1.0 - np.exp(-safe * L)) / safe)
    return np.exp(-0.5 * a_out * L) * core


def _long_integral_quad(a_i, a_j, a_k, a_out, dbeta, L, n=None):
    """Direct quadrature reference.  The sampling density is set from the phase
    mismatch so that the oscillation exp(i dbeta z) is fully resolved."""
    if n is None:
        cycles = abs(dbeta) * L / (2 * np.pi)
        n = int(min(max(4001, 300 * cycles), 40_000_001))
    z = np.linspace(0.0, L, n)
    amp = np.exp(-0.5 * (a_i + a_j + a_k) * z) * np.exp(1j * dbeta * z) \
        * np.exp(-0.5 * a_out * (L - z))
    return np.trapezoid(amp, z)


def degeneracy_factor(i_eq_j, convention="conventional_gamma"):
    """Amplitude degeneracy for pump pairs enumerated with i <= j.

    Expanding dA/dz = i gamma |A|^2 A for a multi-frequency field with the
    CONVENTIONAL scalar coefficient gamma = 2 pi n2 / (lambda A_eff) gives an
    amplitude coefficient 1 for the degenerate pair (i == j) and 2 for a
    distinct pair (i != j) once each unordered pair is counted once.  The
    alternative bookkeeping d = 3, 6 counts the same terms with gamma/3.
    Applying d = 3, 6 together with the conventional gamma overcounts the
    generated AMPLITUDE by 3 and the power by 9; that was the defect in the
    earlier implementation (see CHANGES.md).
    """
    if convention == "conventional_gamma":
        return 1.0 if i_eq_j else 2.0
    if convention == "gamma_over_three":
        return 3.0 if i_eq_j else 6.0
    raise ValueError(f"unknown degeneracy convention {convention!r}")


def enumerate_products(dep, disp, atten, L_m, gamma_per_w_m,
                       drop_pump_coincident=True, quad_check=0,
                       convention="conventional_gamma"):
    """Enumerate nontrivial triplets with pump ordering i <= j and return the
    aggregated product lines.

    Triplets with k == i or k == j are excluded: their generated frequency
    coincides with a pump and they describe SPM/XPM sidebands rather than a new
    FWM tone.  All modeled waves co-propagate, so signed wavevectors reduce to
    the ordinary phase mismatch; the signed form is used explicitly.
    """
    f = dep.freqs_hz
    P0 = dep.launch_w
    n = len(f)
    lam_nm = 299792458.0 / f * 1e9
    a = atten.a_per_m(lam_nm)
    beta = disp.beta(f)

    idx_i, idx_j, idx_k, fo, deg = [], [], [], [], []
    for i in range(n):
        for j in range(i, n):
            for k in range(n):
                if drop_pump_coincident and (k == i or k == j):
                    continue
                idx_i.append(i); idx_j.append(j); idx_k.append(k)
                fo.append(f[i] + f[j] - f[k])
                deg.append(degeneracy_factor(i == j, convention))
    idx_i = np.array(idx_i); idx_j = np.array(idx_j); idx_k = np.array(idx_k)
    fo = np.array(fo); deg = np.array(deg)

    lam_o_nm = 299792458.0 / fo * 1e9
    inside = (lam_o_nm >= atten.valid_nm[0]) & (lam_o_nm <= atten.valid_nm[1])
    a_out = np.where(inside, atten.a_per_m(np.clip(lam_o_nm, *atten.valid_nm)), np.nan)
    n_outside = int(np.count_nonzero(~inside))
    a_out = np.where(inside, a_out, atten.a_per_m(atten.valid_nm[1]))

    beta_o = disp.beta(fo)
    # signed wavevectors, all s = +1 (co-propagating pumps and product)
    dbeta = beta[idx_i] + beta[idx_j] - beta[idx_k] - beta_o

    a_eff = 0.5 * (a[idx_i] + a[idx_j] + a[idx_k])
    core = _long_integral_closed(a_eff, dbeta, L_m, a_out)
    g_eff = (gamma_per_w_m if convention == "conventional_gamma"
             else gamma_per_w_m / 3.0)
    amp = 1j * deg * g_eff * np.sqrt(P0[idx_i] * P0[idx_j] * P0[idx_k]) * core

    audit = {"n_triplets": int(len(fo)),
             "n_product_frequencies": int(len(np.unique(fo))),
             "products_outside_attenuation_validity": n_outside,
             "degeneracy_convention": (
                 "i<=j; d=1 if i==j else 2 with the conventional gamma; k!=i,j"
                 if convention == "conventional_gamma" else
                 "i<=j; d=3 if i==j else 6 with gamma/3; k!=i,j"),
             "gamma_convention": convention,
             "gamma_effective_per_w_m": float(g_eff),
             "direction": "forward generation only; not collected on a return pass"}

    if quad_check:
        rng = np.random.default_rng(12345)
        sel = rng.choice(len(fo), size=min(quad_check, len(fo)), replace=False)
        rel = []
        for t in sel:
            q = _long_integral_quad(a[idx_i[t]], a[idx_j[t]], a[idx_k[t]],
                                    a_out[t], dbeta[t], L_m)
            rel.append(abs(q - core[t]) / max(abs(q), 1e-300))
        audit["closed_vs_quadrature_max_rel"] = float(np.max(rel))
        audit["closed_vs_quadrature_n"] = int(len(sel))

    uf, inv = np.unique(fo, return_inverse=True)
    p_inc = np.bincount(inv, weights=np.abs(amp) ** 2, minlength=len(uf))
    coh_re = np.bincount(inv, weights=amp.real, minlength=len(uf))
    coh_im = np.bincount(inv, weights=amp.imag, minlength=len(uf))
    p_coh = coh_re ** 2 + coh_im ** 2
    return FWMProducts(freq_hz=uf, power_incoherent_w=p_inc,
                       power_coherent_w=p_coh, n_triplets=len(fo),
                       n_lines=len(uf), audit=audit)
