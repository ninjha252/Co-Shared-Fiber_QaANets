#!/usr/bin/env python
"""Guard-band geometry and tuning-margin study.

With 100 GHz classical spacing and a 50 GHz guard on either side, the allowed
interval between two adjacent carriers has width 100 - 2*50 = 0 GHz.  Such
candidates exist only because the guard rule is `>=` at exactly 50 GHz and carry
NO tuning margin.  This script sweeps the guard value and imposes an explicit
frequency-tuning margin, and reports how the optimum and the certification
survive.
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
import protocol as pr
import reliability as rl
import runio as io
from units import hz_to_nm

N, TT = 7, 1


def margin_mask(f_grid, dep, guard_hz, margin_hz):
    """A candidate is retained only if the whole band [f-margin, f+margin] clears
    every carrier by at least `guard_hz`."""
    d = np.abs(np.asarray(f_grid, float)[:, None] - dep.freqs_hz[None, :])
    return np.min(d, axis=1) >= guard_hz + margin_hz - 1e-6


def main():
    t0 = time.time()
    cfg = cf.load("illustrative_baseline.yaml")
    run_id = os.environ.get("QCFC_RUN_ID", "illustrative_baseline")
    d = io.run_dir(run_id)
    man = io.Manifest(run_id, "guard-band geometry and tuning margin", cfg)
    tg = cfg["targets"]
    F_min, Y_min, e0 = tg["F_min"], tg["Y_min"], cfg["link"]["e0_per_pass"]

    GUARDS = (49.0, 50.0, 51.0)
    MARGINS = (0.0, 1.0, 2.0, 2.5, 5.0)   # 2.5 GHz: datasheet laser stability

    dep = cf.build_deployment(cfg)
    # The candidate grid must carry the exact boundary points of EVERY
    # (guard, margin) combination examined, not only the configured one.
    import deployment as dpl
    from units import nm_to_hz
    lo_nm, hi_nm = cfg["search"]["hardware_domain_nm"]
    f_lo, f_hi = nm_to_hz(hi_nm), nm_to_hz(lo_nm)
    extra = [f_lo, f_hi] + list(dep.freqs_hz)
    for g_ in GUARDS:
        for m_ in MARGINS:
            extra += list(dpl.guard_boundaries(dep, (g_ + m_) * 1e9))
    extra = [e for e in extra if f_lo <= e <= f_hi]
    grid = dpl.candidate_grid(f_lo, f_hi, cfg["search"]["grid_step_ghz"] * 1e9,
                              extra)
    hw, _, se, _, _ = cf.build_masks(cfg, grid, dep)
    lam = hz_to_nm(grid)

    # geometry record
    fs = np.sort(dep.freqs_hz) / 1e9
    gaps = np.diff(fs)
    geo = [dict(quantity="classical spacings present (GHz)",
                value=str(sorted(set(np.round(gaps, 3).tolist())))),
           dict(quantity="adjacent 100 GHz carrier pairs", value=int((gaps == 100).sum())),
           dict(quantity="allowed width between adjacent carriers at 50 GHz guard (GHz)",
                value=100 - 2 * 50),
           dict(quantity="vacant grid slots inside the hardware domain",
                value=str(sorted(set(np.arange(191600, 195901, 100).tolist())
                                 - set(fs.astype(int).tolist())))),
           dict(quantity="note",
                value="between-carrier candidates are zero-width isolated points that "
                      "exist only under a >= rule at exactly 50 GHz; they carry no "
                      "tuning margin")]
    man.add_output(io.write_csv(os.path.join(d, "guard_geometry.csv"), geo))

    # cache the physics once per operating point (guard/margin only change the mask)
    cache = {}
    for L, P in [(10.0, -5.0), (5.0, -10.0), (3.0, -13.0), (2.0, -15.0)]:
        sc = cf.build_scenario(cfg, length_km=L, launch_dbm=P)
        eta = np.zeros((len(grid), 3)); nu = np.zeros((len(grid), 3))
        sg = np.zeros((len(grid), 3))
        for i, f in enumerate(grid):
            r = ol.evaluate_candidate(sc, f, want_components=False)
            eta[i] = r["eta"]; nu[i] = r["nu"]; sg[i] = r["sigma2_phi"]
        T, nuf, _ = oc.compose_passes(eta, nu)
        res = oc.accepted_optical(T, nuf)
        z = lp.z_from_sigma2(sg)
        b = np.stack([lp.traversal_weights(e0, z[..., m])[..., 0] for m in range(3)],
                     axis=-1)
        q = lp.affected_probability(res.wI, b)
        cache[(L, P)] = dict(s=res.s, q=q, C=rl.correctable_weight(q, N, TT),
                             eta=eta, nu_pass=nu, z=z, nu=nuf)
        print(f"  physics cached for {L} km, {P} dBm ({time.time()-t0:.0f} s)", flush=True)

    keys_id = pr.key_set("identity")
    rows = []
    for (L, P), c in cache.items():
        Ylb = c["s"] ** N * c["C"]
        for guard_ghz in GUARDS:
            for margin_ghz in MARGINS:
                allowed = hw & se & margin_mask(grid, dep, guard_ghz * 1e9,
                                                margin_ghz * 1e9)
                mF = allowed & (c["s"] > 0) & (c["C"] >= F_min)
                iv = opt.intervals_from_mask(lam, allowed)
                widths = [hi - lo for lo, hi in iv]
                base = dict(length_km=L, launch_dbm=P, guard_ghz=guard_ghz,
                            tuning_margin_ghz=margin_ghz,
                            n_allowed=int(allowed.sum()),
                            n_connected_components=len(iv),
                            n_zero_width_components=sum(1 for w in widths if w < 1e-9),
                            max_component_width_nm=(max(widths) if widths else 0.0),
                            n_Omega_F=int(mF.sum()))
                if mF.any():
                    k = int(np.nonzero(mF)[0][np.argmax(Ylb[mF])])
                    p1 = pr.channel_point(c["eta"][k], c["nu_pass"][k], e0, c["z"][k])
                    fe, _ = pr.exact_fidelity(p1, keys_id, "pauli_enumeration")
                    mW = mF & (Ylb >= Y_min)
                    base.update(lambda_star_nm=float(lam[k]),
                                frequency_star_ghz=float(grid[k] / 1e9),
                                C_bound_at_star=float(c["C"][k]),
                                s_at_star=float(c["s"][k]),
                                P_acc_at_star=float(c["s"][k] ** N),
                                Y_lb_at_star=float(Ylb[k]),
                                Fe_exact_at_star=fe,
                                nu_FWM_share_at_star=float(
                                    ol.evaluate_candidate(
                                        cf.build_scenario(cfg, length_km=L,
                                                          launch_dbm=P),
                                        grid[k], want_components=False)["nu_FWM"].sum()
                                    / max(c["nu"][k], 1e-300)),
                                W_q_nonempty=bool(mW.any()),
                                status=("certified" if mW.any() else
                                        "quality certified; yield not certified"))
                else:
                    base.update(lambda_star_nm=np.nan, frequency_star_ghz=np.nan,
                                C_bound_at_star=np.nan, s_at_star=np.nan,
                                P_acc_at_star=np.nan, Y_lb_at_star=np.nan,
                                Fe_exact_at_star=np.nan, nu_FWM_share_at_star=np.nan,
                                W_q_nonempty=False,
                                status="NOT certified: Omega_F empty")
                rows.append(base)
    # Wavelength displacement and relative yield change against the SAME guard
    # at zero added margin.  A ratio is computed only where BOTH optima exist.
    ref = {(r["length_km"], r["launch_dbm"], r["guard_ghz"]): r
           for r in rows if r["tuning_margin_ghz"] == 0.0}
    for r in rows:
        b = ref.get((r["length_km"], r["launch_dbm"], r["guard_ghz"]))
        both = (b is not None and np.isfinite(r["Y_lb_at_star"])
                and np.isfinite(b["Y_lb_at_star"]) and b["Y_lb_at_star"] > 0)
        r["reference_rule"] = "same guard, zero added margin"
        r["delta_lambda_nm_vs_zero_margin"] = (
            r["lambda_star_nm"] - b["lambda_star_nm"] if both else np.nan)
        r["relative_Y_lb_change_vs_zero_margin"] = (
            r["Y_lb_at_star"] / b["Y_lb_at_star"] - 1.0 if both else np.nan)
        r["both_optima_exist"] = bool(both)
    man.add_output(io.write_csv(os.path.join(d, "guard_margin_study.csv"), rows))
    man.note("With 100 GHz spacing and a 50 GHz guard the between-carrier allowed "
             "width is exactly zero; candidates there have no tuning margin.")
    man.save(os.path.join(d, "run_manifest_guard.json"))

    print(f"\nguard/margin study: {len(rows)} rows ({time.time()-t0:.0f} s)")
    for r in rows:
        if r["length_km"] in (5.0, 3.0):
            print(f"  L={r['length_km']:.0f} P={r['launch_dbm']:.0f}  guard={r['guard_ghz']:.0f} "
                  f"margin={r['tuning_margin_ghz']:.0f}  allowed={r['n_allowed']:4d} "
                  f"comps={r['n_connected_components']:3d} zero-w={r['n_zero_width_components']:3d} "
                  f"nOmF={r['n_Omega_F']:4d}  lam*={r['lambda_star_nm'] if not np.isnan(r['lambda_star_nm']) else float('nan'):9.3f} "
                  f"C={r['C_bound_at_star'] if not np.isnan(r['C_bound_at_star']) else float('nan'):8.5f} "
                  f"{r['status']}")


if __name__ == "__main__":
    main()
