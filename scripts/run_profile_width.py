#!/usr/bin/env python
"""FWM spectral-profile diagnostic (Methods: four-wave mixing).

The production model treats every generated FWM component as a delta line even
though the classical spectral model carries a finite modulation width.  This
script compares the narrow-line limit with PRESCRIBED Gaussian profiles of 1, 5
and 25 GHz rms at the predeclared operating points and reports:

  * the peak propagated FWM occupancy over the allowed domain,
  * the FWM occupancy at the selected wavelength, which sits away from any exact
    mixing frequency, so it is fed only by the profile tails,
  * the size of the certified set, the selected wavelength and its yield.

The profiles are phenomenological broadenings at fixed total product power.
They are NOT a simulation of modulated four-wave mixing: the correct treatment
needs the field statistics and intensity moments of the modulated carriers, and
distinct-channel and degenerate processes differ.  The widths are prescribed,
not derived from a modulation model.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import config as cf
import local_protocol as lp
import optical_channel as oc
import optical_link as ol
import optimization as opt
import reliability as rl
import runio as io
from units import hz_to_nm

N, TT = 7, 1
WIDTHS_GHZ = (0.0, 1.0, 5.0, 25.0)
POINTS = [(2.0, -15.0), (3.0, -13.0), (5.0, -10.0), (10.0, -5.0)]


def solve(cfg, grid, allowed, L, P, width_ghz, e0, F_min, Y_min):
    sc = cf.build_scenario(cfg, length_km=L, launch_dbm=P,
                           overrides={"fwm_profile_rms_hz": width_ghz * 1e9})
    n = len(grid)
    eta = np.zeros((n, 3)); nu = np.zeros((n, 3)); sg = np.zeros((n, 3))
    nuF = np.zeros(n)
    for i, f in enumerate(grid):
        r = ol.evaluate_candidate(sc, f, want_components=False)
        eta[i] = r["eta"]; nu[i] = r["nu"]; sg[i] = r["sigma2_phi"]
        nuF[i] = float(np.sum(r["nu_FWM"]))
    T, nuf, _ = oc.compose_passes(eta, nu)
    res = oc.accepted_optical(T, nuf)
    z = lp.z_from_sigma2(sg)
    b = np.stack([lp.traversal_weights(e0, z[..., m])[..., 0] for m in range(3)],
                 axis=-1)
    q = lp.affected_probability(res.wI, b)
    C = rl.correctable_weight(q, N, TT)
    Ylb = res.s ** N * C
    mF = allowed & (res.s > 0) & (C >= F_min)
    out = dict(length_km=L, launch_dbm=P, fwm_profile_rms_ghz=width_ghz,
               n_allowed=int(allowed.sum()), n_Omega_F=int(mF.sum()),
               max_nu_FWM_over_allowed=float(np.max(nuF[allowed])),
               median_nu_FWM_over_allowed=float(np.median(nuF[allowed])),
               max_C_over_allowed=float(np.max(C[allowed])))
    if mF.any():
        k = int(np.nonzero(mF)[0][np.argmax(Ylb[mF])])
        mW = mF & (Ylb >= Y_min)
        out.update(lambda_star_nm=float(hz_to_nm(grid[k])),
                   s_at_star=float(res.s[k]), q_at_star=float(q[k]),
                   C_at_star=float(C[k]), Y_lb_at_star=float(Ylb[k]),
                   nu_FWM_at_star=float(nuF[k]),
                   nu_total_at_star=float(nuf[k]),
                   fwm_share_of_nu_at_star=float(nuF[k] / nuf[k])
                   if nuf[k] > 0 else np.nan,
                   W_q_nonempty=bool(mW.any()),
                   status="certified" if mW.any() else
                          "quality certified; yield not certified")
    else:
        out.update(lambda_star_nm=np.nan, s_at_star=np.nan, q_at_star=np.nan,
                   C_at_star=np.nan, Y_lb_at_star=np.nan, nu_FWM_at_star=np.nan,
                   nu_total_at_star=np.nan, fwm_share_of_nu_at_star=np.nan,
                   W_q_nonempty=False, status="NOT certified: Omega_F empty")
    return out


def main():
    t0 = time.time()
    cfg = cf.load("illustrative_baseline.yaml")
    run_id = os.environ.get("QCFC_RUN_ID", "illustrative_baseline")
    d = io.run_dir(run_id)
    man = io.Manifest(run_id, "FWM spectral-profile diagnostic", cfg)
    tg = cfg["targets"]
    F_min, Y_min, e0 = tg["F_min"], tg["Y_min"], cfg["link"]["e0_per_pass"]

    dep = cf.build_deployment(cfg)
    grid = cf.build_grid(cfg, dep)
    _, _, _, allowed, _ = cf.build_masks(cfg, grid, dep)

    rows = []
    for L, P in POINTS:
        for wg in WIDTHS_GHZ:
            r = solve(cfg, grid, allowed, L, P, wg, e0, F_min, Y_min)
            r["profile_status"] = (
                "analytic narrow-line limit" if wg == 0 else
                "PRESCRIBED Gaussian profile at fixed product power; not a "
                "modulated-FWM simulation")
            rows.append(r)
            print(f"  L={L:4.1f} km P={P:+.0f} dBm width={wg:5.1f} GHz  "
                  f"nOmF={r['n_Omega_F']:4d}  lam*="
                  f"{r['lambda_star_nm']:9.3f}  Y_lb={r['Y_lb_at_star']:.4e}  "
                  f"maxnuFWM={r['max_nu_FWM_over_allowed']:.3e} "
                  f"({time.time()-t0:.0f} s)", flush=True)

    # changes relative to the delta-line limit at the same operating point
    ref = {(r["length_km"], r["launch_dbm"]): r
           for r in rows if r["fwm_profile_rms_ghz"] == 0.0}
    for r in rows:
        b = ref[(r["length_km"], r["launch_dbm"])]
        both = np.isfinite(r["Y_lb_at_star"]) and np.isfinite(b["Y_lb_at_star"]) \
            and b["Y_lb_at_star"] > 0
        r["reference_rule"] = "same operating point, narrow-line limit"
        r["delta_lambda_nm_vs_delta_line"] = (
            r["lambda_star_nm"] - b["lambda_star_nm"] if both else np.nan)
        r["relative_Y_lb_change_vs_delta_line"] = (
            r["Y_lb_at_star"] / b["Y_lb_at_star"] - 1.0 if both else np.nan)
        r["n_Omega_F_change_vs_delta_line"] = r["n_Omega_F"] - b["n_Omega_F"]
        r["peak_nu_FWM_ratio_vs_delta_line"] = (
            r["max_nu_FWM_over_allowed"] / b["max_nu_FWM_over_allowed"]
            if b["max_nu_FWM_over_allowed"] > 0 else np.nan)
    man.add_output(io.write_csv(os.path.join(d, "fwm_profile_width.csv"), rows))
    man.note("Prescribed profile widths; the narrow-line result is the width -> 0 "
             "limit of the same expression.")
    man.save(os.path.join(d, "run_manifest_profile.json"))
    print(f"\nFWM profile diagnostic: {len(rows)} rows ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
