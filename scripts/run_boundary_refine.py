#!/usr/bin/env python
"""Refine the certification boundary of the operating map.

On the 1 km x 1 dB map the boundary is a staircase artifact of the grid.  The
certified domain is nonempty exactly when

    M(L, P) = max_{lambda in Omega_q} C_7,1(q(L, P, lambda)) >= F_min,

so the boundary at fixed power is a root of M(L, P) - F_min.  The root is found
with a BRACKETED bisection on directly re-evaluated M, never by interpolating
stored values, and the wavelength maximization inside each evaluation runs over
the full allowed candidate set.  Monotonicity is not assumed: the stored map is
scanned for every sign change, and a power showing more than one crossing is
reported as such instead of being reduced to a single boundary.  The reported
target is a residual of 1e-6 in C; a bracket that does not reach it is reported
as unresolved.
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import config as cf
import local_protocol as lp
import optical_channel as oc
import optical_link as ol
import reliability as rl
import runio as io

N, TT = 7, 1
C_RESIDUAL_TARGET = 1e-6
MAX_BISECTIONS = 40


def max_C(cfg, f_allowed, L, P, e0, return_C=False):
    sc = cf.build_scenario(cfg, length_km=float(L), launch_dbm=float(P))
    eta = np.zeros((len(f_allowed), 3)); nu = np.zeros((len(f_allowed), 3))
    sg = np.zeros((len(f_allowed), 3))
    for i, f in enumerate(f_allowed):
        r = ol.evaluate_candidate(sc, f, want_components=False)
        eta[i] = r["eta"]; nu[i] = r["nu"]; sg[i] = r["sigma2_phi"]
    T, nuf, _ = oc.compose_passes(eta, nu)
    res = oc.accepted_optical(T, nuf)
    z = lp.z_from_sigma2(sg)
    b = np.stack([lp.traversal_weights(e0, z[..., m])[..., 0] for m in range(3)], axis=-1)
    C = rl.correctable_weight(lp.affected_probability(res.wI, b), N, TT)
    return (float(np.max(C)), C) if return_C else float(np.max(C))


def top_k_subset(cfg, f_allowed, L, P, e0, k=32):
    """Candidates holding the largest C at a bracket midpoint.

    The wavelength maximization inside the root solve runs over this subset, and
    the result is VERIFIED against the full allowed set at the located root; if
    the two maxima disagree the subset is widened and the solve repeated.
    """
    _, C = max_C(cfg, f_allowed, L, P, e0, return_C=True)
    idx = np.argsort(C)[::-1][:k]
    return np.sort(f_allowed[idx])


def bisect_boundary(fun, lo, hi, f_lo, f_hi, target, max_iter=MAX_BISECTIONS):
    """Bisection on a sign-changing bracket of `fun - target`.  Returns
    (root, value_at_root, n_evaluations, status)."""
    n = 0
    a, b, fa, fb = lo, hi, f_lo - target, f_hi - target
    if fa == 0.0:
        return lo, f_lo, 0, "exact at the lower bracket"
    if fb == 0.0:
        return hi, f_hi, 0, "exact at the upper bracket"
    if fa * fb > 0:
        return np.nan, np.nan, 0, "no sign change in the supplied bracket"
    mid, fmid = np.nan, np.nan
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = fun(mid) - target
        n += 1
        if abs(fmid) <= C_RESIDUAL_TARGET:
            return mid, fmid + target, n, "converged"
        if fa * fmid < 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid
    return mid, fmid + target, n, (f"UNRESOLVED: |M - F_min| = {abs(fmid):.2e} "
                                   f"after {n} evaluations")


def main():
    t0 = time.time()
    cfg = cf.load("illustrative_baseline.yaml")
    d = io.run_dir(os.environ.get("QCFC_RUN_ID", "illustrative_baseline"))
    man = io.Manifest("illustrative_baseline", "certification boundary refinement", cfg)
    F_min = cfg["targets"]["F_min"]; e0 = cfg["link"]["e0_per_pass"]
    dep = cf.build_deployment(cfg)
    grid = cf.build_grid(cfg, dep)
    _, _, _, allowed, _ = cf.build_masks(cfg, grid, dep)
    f_allowed = grid[allowed]

    m = pd.read_csv(os.path.join(d, "operating_map.csv"))
    rows = []
    for P, g in m.groupby("launch_dbm"):
        g = g.sort_values("length_km")
        L = g["length_km"].to_numpy(float)
        C = g["max_C_bound_over_Omega_q"].to_numpy(float)
        above = C >= F_min
        crossings = [i for i in range(1, len(L)) if above[i - 1] != above[i]]
        if not crossings:
            rows.append(dict(launch_dbm=float(P), boundary_length_km=np.nan,
                             method="direct re-evaluation; bracketed bisection",
                             n_sign_changes_on_the_sampled_grid=0,
                             grid_lower_km=np.nan, grid_upper_km=np.nan,
                             n_boundary_evaluations=0, max_C_at_boundary=np.nan,
                             residual_in_C=np.nan,
                             status=("no crossing on the sampled 1-30 km grid: "
                                     + ("certified throughout" if above.all()
                                        else "uncertified throughout"))))
            continue
        for ci, k in enumerate(crossings):
            lo, hi = float(L[k - 1]), float(L[k])
            ksub, verified = 32, False
            while not verified and ksub <= 256:
                sub = top_k_subset(cfg, f_allowed, 0.5 * (lo + hi), float(P), e0,
                                   k=ksub)
                fun = lambda x: max_C(cfg, sub, x, float(P), e0)
                root, val, nev, status = bisect_boundary(
                    fun, lo, hi, float(C[k - 1]), float(C[k]), F_min)
                if not np.isfinite(root):
                    break
                full = max_C(cfg, f_allowed, root, float(P), e0)
                verified = abs(full - val) <= 1e-12
                if not verified:
                    ksub *= 2
            if np.isfinite(root):
                val = full
                status = (status if verified else
                          f"{status}; subset maximization NOT verified against "
                          f"the full allowed set (|difference| = "
                          f"{abs(full - val):.2e})")
                if abs(val - F_min) > C_RESIDUAL_TARGET and status == "converged":
                    status = (f"UNRESOLVED after full-domain verification: "
                              f"|M - F_min| = {abs(val - F_min):.2e}")
            rows.append(dict(
                launch_dbm=float(P), boundary_length_km=root,
                method="direct re-evaluation; bracketed bisection",
                n_sign_changes_on_the_sampled_grid=len(crossings),
                grid_lower_km=lo, grid_upper_km=hi,
                n_boundary_evaluations=nev,
                lambda_subset_size=int(ksub),
                subset_maximization_verified_against_full_domain=bool(verified),
                max_C_at_boundary=val,
                residual_in_C=(abs(val - F_min) if np.isfinite(val) else np.nan),
                status=(status if len(crossings) == 1 else
                        f"{status}; crossing {ci+1} of {len(crossings)} at this "
                        f"power: monotonicity in length is NOT assumed")))
            print(f"  P={P:+.0f} dBm: boundary L={root:.4f} km, max C={val:.8f}, "
                  f"residual {abs(val-F_min):.1e} after {nev} evaluations "
                  f"({time.time()-t0:.0f} s)", flush=True)
    man.add_output(io.write_csv(os.path.join(d, "certification_boundary.csv"), rows))
    res = [r["residual_in_C"] for r in rows if np.isfinite(r.get("residual_in_C", np.nan))]
    worst = max(res) if res else np.nan
    man.tolerance("boundary_max_residual_in_C", float(worst))
    man.tolerance("boundary_residual_target_in_C", C_RESIDUAL_TARGET)
    man.note("Boundary located by bracketed bisection on directly re-evaluated "
             "max_lambda C; the stored map is used only to supply brackets, and "
             "every sign change found on the grid is solved separately.")
    man.save(os.path.join(d, "run_manifest_boundary.json"))
    nun = sum(1 for r in rows if str(r["status"]).startswith("UNRESOLVED"))
    print(f"\nboundary refinement: {len(rows)} rows, worst residual "
          f"{worst:.2e} in C, {nun} unresolved ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
