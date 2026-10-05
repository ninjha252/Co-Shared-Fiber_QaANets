"""Wavelength scan, feasible sets, selectors, and certified regions
(Methods: selection rules)."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

import optical_channel as oc
import local_protocol as lp
import reliability as rl
import optical_link as ol
from units import C_LIGHT


def scan_wavelengths(sc, f_grid, e0_per_pass=None, n=7, t=1, progress=None):
    """Evaluate every candidate frequency and return a dict of stacked arrays."""
    e0 = sc.e0_per_pass if e0_per_pass is None else e0_per_pass
    f_grid = np.asarray(f_grid, dtype=float)
    m = len(f_grid)
    cols = {k: np.zeros(m) for k in
            ["eta_1", "eta_2", "eta_3", "T",
             "nu_R_1", "nu_R_2", "nu_R_3", "nu_FWM_1", "nu_FWM_2", "nu_FWM_3",
             "nu_leak_1", "nu_leak_2", "nu_leak_3", "nu_1", "nu_2", "nu_3",
             "nu_R_final", "nu_FWM_final", "nu_leak_final", "nu_final",
             "sigma2_phi_1", "sigma2_phi_2", "sigma2_phi_3",
             "z_1", "z_2", "z_3", "b_1", "b_2", "b_3",
             "c0", "c1", "s", "r", "wI", "q", "C_bound", "P_acc",
             "Y_lb", "Y_ub", "log_Y_lb", "log_Y_ub", "T_filter_signal"]}
    for i, f in enumerate(f_grid):
        if progress and i % progress == 0:
            print(f"  ... {i}/{m}", flush=True)
        rr = ol.evaluate_candidate(sc, f, want_components=False)
        eta, nu_pass = rr["eta"], rr["nu"]
        T, nu, prop = oc.compose_passes(eta, nu_pass)
        w = oc.propagation_weights(eta)
        res = oc.accepted_optical(T, nu)
        z = lp.z_from_sigma2(rr["sigma2_phi"])
        tw = [lp.traversal_weights(e0, zz) for zz in z]
        b = np.array([x[0] for x in tw])
        q = float(lp.affected_probability(res.wI, b))
        C = float(rl.correctable_weight(q, n, t))
        s = float(res.s)
        Pacc = s ** n
        for j in range(3):
            cols[f"eta_{j+1}"][i] = eta[j]
            cols[f"nu_R_{j+1}"][i] = rr["nu_R"][j]
            cols[f"nu_FWM_{j+1}"][i] = rr["nu_FWM"][j]
            cols[f"nu_leak_{j+1}"][i] = rr["nu_leak"][j]
            cols[f"nu_{j+1}"][i] = nu_pass[j]
            cols[f"sigma2_phi_{j+1}"][i] = rr["sigma2_phi"][j]
            cols[f"z_{j+1}"][i] = z[j]
            cols[f"b_{j+1}"][i] = b[j]
        cols["T"][i] = T
        cols["nu_R_final"][i] = float(np.sum(w * rr["nu_R"]))
        cols["nu_FWM_final"][i] = float(np.sum(w * rr["nu_FWM"]))
        cols["nu_leak_final"][i] = float(np.sum(w * rr["nu_leak"]))
        cols["nu_final"][i] = nu
        cols["c0"][i] = float(res.c0); cols["c1"][i] = float(res.c1)
        cols["s"][i] = s; cols["r"][i] = float(res.r); cols["wI"][i] = float(res.wI)
        cols["q"][i] = q; cols["C_bound"][i] = C
        cols["P_acc"][i] = Pacc
        cols["Y_lb"][i] = Pacc * C; cols["Y_ub"][i] = Pacc
        cols["log_Y_lb"][i] = n * np.log(s) + np.log(C) if (s > 0 and C > 0) else -np.inf
        cols["log_Y_ub"][i] = n * np.log(s) if s > 0 else -np.inf
        cols["T_filter_signal"][i] = rr["T_filter_signal"]
    cols["frequency_hz"] = f_grid
    cols["wavelength_nm"] = C_LIGHT / f_grid * 1e9
    return cols


def intervals_from_mask(x, mask):
    """Contiguous [x_lo, x_hi] runs of True in `mask`, as grid-endpoint spans."""
    x = np.asarray(x); mask = np.asarray(mask, dtype=bool)
    out = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((float(min(x[i], x[j])), float(max(x[i], x[j]))))
            i = j + 1
        else:
            i += 1
    return out


def refine_crossing(fun, x_lo, x_hi, target, xtol=1e-6):
    """Bracketed root of fun(x) = target, returning None if not bracketed."""
    f_lo, f_hi = fun(x_lo) - target, fun(x_hi) - target
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return None
    return float(brentq(lambda z: fun(z) - target, x_lo, x_hi, xtol=xtol))


@dataclass
class SelectionResult:
    rule: str
    index: int
    wavelength_nm: float
    searched_over: str


def selectors(cols, allowed, in_omega_f):
    """Comparator selection rules, all evaluated on the same physical model.

    Each rule is reported with the domain it searched: `Omega_q` is every
    allowed candidate, `Omega_F` only those meeting the fidelity certificate.
    Rules that search different domains are not interchangeable, so the domain
    is carried through to the output table rather than being implied.
    """
    out = {}
    idx_all = np.nonzero(allowed)[0]
    idx_f = np.nonzero(in_omega_f)[0]

    def best(idx, key, sign=1.0, label="", dom=""):
        if len(idx) == 0:
            return None
        vals = sign * cols[key][idx]
        k = idx[int(np.nanargmax(vals))]
        return SelectionResult(label, int(k), float(cols["wavelength_nm"][k]), dom)

    out["max_Ylb_subject_to_Fmin"] = best(idx_f, "Y_lb", 1.0,
                                          "max Y_lb subject to F_min", "Omega_F")
    out["max_T_over_Omega_q"] = best(idx_all, "T", 1.0,
                                     "max T (loss-only)", "Omega_q")
    out["max_T_over_Omega_F"] = best(idx_f, "T", 1.0,
                                     "max T (loss-only)", "Omega_F")
    out["min_q_over_Omega_q"] = best(idx_all, "q", -1.0,
                                     "min q (quality-oriented)", "Omega_q")
    out["min_q_over_Omega_F"] = best(idx_f, "q", -1.0,
                                     "min q (quality-oriented)", "Omega_F")
    return out


def fixed_reference_index(cols, allowed, target_nm, tol_nm=1e-6):
    """Exact fixed-reference rule.

    Returns (index, status).  The predeclared reference wavelength is a valid
    selection ONLY if it is itself an allowed candidate; otherwise the rule is
    reported as invalid at this deployment.  Snapping to a neighbour is a
    DIFFERENT rule (`nearest_allowed_index`), never a repair of this one.
    """
    idx = np.nonzero(allowed)[0]
    if len(idx) == 0:
        return None, "undefined: no allowed candidate"
    k = idx[int(np.argmin(np.abs(cols["wavelength_nm"][idx] - target_nm)))]
    if abs(float(cols["wavelength_nm"][k]) - target_nm) <= tol_nm:
        return int(k), "valid: the reference wavelength is an allowed candidate"
    return None, ("INVALID: the reference wavelength is excluded by the guard "
                  "and margin rule at this deployment")


def nearest_allowed_index(cols, allowed, target_nm):
    """Distinct comparator: the allowed candidate nearest to the reference."""
    idx = np.nonzero(allowed)[0]
    if len(idx) == 0:
        return None
    return int(idx[int(np.argmin(np.abs(cols["wavelength_nm"][idx] - target_nm)))])
