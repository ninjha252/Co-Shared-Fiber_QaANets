#!/usr/bin/env python
"""Stage E: length/power operating map with the quantum wavelength re-optimized
at every point (Methods: operating map)."""
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
from protocol import certificate_threshold_q, e0_max_for_certification
import reliability as rl
import runio as io
from units import dbm_to_w

N, T_CODE = 7, 1


def components_at_length(sc, f_allowed):
    """Per-candidate optical components at the scenario's reference power."""
    out = {k: np.zeros((len(f_allowed), 3)) for k in
           ("eta", "nu_R", "nu_FWM", "nu_leak", "sigma2")}
    for i, f in enumerate(f_allowed):
        r = ol.evaluate_candidate(sc, f, want_components=False)
        out["eta"][i] = r["eta"]; out["nu_R"][i] = r["nu_R"]
        out["nu_FWM"][i] = r["nu_FWM"]; out["nu_leak"][i] = r["nu_leak"]
        out["sigma2"][i] = r["sigma2_phi"]
    return out


def rescale(comp, kappa):
    """Undepleted-pump power scaling: Raman and leakage linear, FWM cubic,
    XPM variance quadratic.  Verified against direct evaluation below."""
    nu = kappa * comp["nu_R"] + kappa ** 3 * comp["nu_FWM"] + kappa * comp["nu_leak"]
    return comp["eta"], nu, kappa ** 2 * comp["sigma2"], \
        kappa * comp["nu_R"], kappa ** 3 * comp["nu_FWM"], kappa * comp["nu_leak"]


def reduce_point(eta, nu_pass, sigma2, e0):
    T, nu, _ = oc.compose_passes(eta, nu_pass)
    res = oc.accepted_optical(T, nu)
    z = lp.z_from_sigma2(sigma2)
    tw = [lp.traversal_weights(e0, z[..., m]) for m in range(3)]
    b = np.stack([w[..., 0] for w in tw], axis=-1)
    q = lp.affected_probability(res.wI, b)
    C = rl.correctable_weight(q, N, T_CODE)
    s = res.s
    return dict(T=T, nu=nu, s=s, r=res.r, q=q, C=C, P_acc=s ** N,
                Y_lb=s ** N * C, Y_ub=s ** N, z=z)


def main():
    t0 = time.time()
    cfg = cf.load("illustrative_baseline.yaml")
    run_id = os.environ.get("QCFC_RUN_ID", cfg["scenario_id"])
    d = io.run_dir(run_id)
    man = io.Manifest(run_id, "E operating map", cfg)
    tg = cfg["targets"]
    F_min, Y_min, eps, e0 = tg["F_min"], tg["Y_min"], tg["epsilon"], cfg["link"]["e0_per_pass"]

    dep = cf.build_deployment(cfg)
    grid = cf.build_grid(cfg, dep)
    _, _, _, allowed, _ = cf.build_masks(cfg, grid, dep)
    f_allowed = grid[allowed]
    allowed_indices = np.flatnonzero(allowed)
    lam_allowed = 299792458.0 / f_allowed * 1e9
    P_ref_dbm = cfg["deployment"]["classical_launch_dbm_per_channel"]
    P_ref = dbm_to_w(P_ref_dbm)

    mp_cfg = cfg.get("map", {})
    lengths = np.arange(float(mp_cfg.get("length_km_min", 1.0)),
                        float(mp_cfg.get("length_km_max", 30.0)) + 0.5,
                        float(mp_cfg.get("length_km_step", 1.0)))
    powers = np.arange(float(mp_cfg.get("launch_dbm_min", -15.0)),
                       float(mp_cfg.get("launch_dbm_max", 0.0)) + 0.5,
                       float(mp_cfg.get("launch_dbm_step", 1.0)))
    q_star = certificate_threshold_q(F_min)

    # ------------------------------------------- verify the power scaling first
    ver = []
    for L in (5.0, 15.0):
        sc = cf.build_scenario(cfg, length_km=L)
        comp = components_at_length(sc, f_allowed[:: max(1, len(f_allowed) // 12)])
        sub = f_allowed[:: max(1, len(f_allowed) // 12)]
        for Pd in (-12.0, -2.0):
            k = float(dbm_to_w(Pd) / P_ref)
            eta_s, nu_s, sig_s, nR, nF, nL = rescale(comp, k)
            sc2 = cf.build_scenario(cfg, length_km=L, launch_dbm=Pd)
            for i, f in enumerate(sub):
                r = ol.evaluate_candidate(sc2, f, want_components=False)
                for nm_, a, b in (("nu_R", nR[i], r["nu_R"]),
                                  ("nu_FWM", nF[i], r["nu_FWM"]),
                                  ("nu_leak", nL[i], r["nu_leak"]),
                                  ("sigma2_phi", sig_s[i], r["sigma2_phi"])):
                    den = np.maximum(np.abs(b), 1e-300)
                    ver.append(dict(length_km=L, launch_dbm=Pd,
                                    wavelength_nm=299792458.0 / f * 1e9,
                                    quantity=nm_,
                                    max_relative_error=float(np.max(np.abs(a - b) / den))))
    worst_scale = max(r["max_relative_error"] for r in ver)
    man.add_output(io.write_csv(os.path.join(d, "power_scaling_verification.csv"), ver))
    man.tolerance("power_scaling_max_relative_error", worst_scale)
    print(f"power-scaling verification: max relative error {worst_scale:.2e}")
    if worst_scale > 1e-10:
        raise SystemExit("power scaling not verified; refusing to use the acceleration")

    # ---------------------------------------------------------------- the map
    rows, summary = [], []
    for L in lengths:
        sc = cf.build_scenario(cfg, length_km=float(L))
        comp = components_at_length(sc, f_allowed)
        for Pd in powers:
            k = float(dbm_to_w(Pd) / P_ref)
            eta_s, nu_s, sig_s, nR, nF, nL = rescale(comp, k)
            pt = reduce_point(eta_s, nu_s, sig_s, e0)
            mF = (pt["s"] > 0) & (pt["C"] >= F_min)
            if mF.any():
                idx = int(np.nonzero(mF)[0][np.argmax(pt["Y_lb"][mF])])
                mW = mF & (pt["Y_lb"] >= Y_min)
                mN = mF & (pt["Y_lb"] >= (1 - eps) * pt["Y_lb"][idx])
                status = ("certified for F_min and Y_min" if mW.any() else
                          "quality certified; yield not certified by the lower bound")
                lam_star = float(lam_allowed[idx])
                row = dict(length_km=float(L), launch_dbm=float(Pd),
                           lambda_star_nm=lam_star,
                           T_at_star=float(pt["T"][idx]), s_at_star=float(pt["s"][idx]),
                           r_at_star=float(pt["r"][idx]), q_at_star=float(pt["q"][idx]),
                           C_bound_at_star=float(pt["C"][idx]),
                           P_acc_at_star=float(pt["P_acc"][idx]),
                           Y_lb_at_star=float(pt["Y_lb"][idx]),
                           Y_ub_at_star=float(pt["Y_ub"][idx]),
                           log10_Y_lb_at_star=float(np.log10(pt["Y_lb"][idx]))
                           if pt["Y_lb"][idx] > 0 else -np.inf,
                           quality_domain_nonempty=True,
                           n_Omega_F=int(mF.sum()), W_q_nonempty=bool(mW.any()),
                           n_W_q=int(mW.sum()),
                           W_q_intervals=len(opt.intervals_from_mask(
                               lam_allowed, mW, allowed_indices)),
                           N_epsilon_intervals=len(opt.intervals_from_mask(
                               lam_allowed, mN, allowed_indices)),
                           n_N_epsilon=int(mN.sum()),
                           max_Y_ub_over_Omega_q=float(np.max(pt["Y_ub"])),
                           max_C_bound_over_Omega_q=float(np.max(pt["C"])),
                           yield_excluded_by_upper_bound=bool(np.max(pt["Y_ub"]) < Y_min),
                           status_reason=status)
            else:
                row = dict(length_km=float(L), launch_dbm=float(Pd),
                           lambda_star_nm=np.nan, T_at_star=np.nan, s_at_star=np.nan,
                           r_at_star=np.nan, q_at_star=np.nan, C_bound_at_star=np.nan,
                           P_acc_at_star=np.nan, Y_lb_at_star=np.nan, Y_ub_at_star=np.nan,
                           log10_Y_lb_at_star=np.nan, quality_domain_nonempty=False,
                           n_Omega_F=0, W_q_nonempty=False, n_W_q=0, W_q_intervals=0,
                           N_epsilon_intervals=0, n_N_epsilon=0,
                           max_Y_ub_over_Omega_q=float(np.max(pt["Y_ub"])),
                           max_C_bound_over_Omega_q=float(np.max(pt["C"])),
                           yield_excluded_by_upper_bound=bool(np.max(pt["Y_ub"]) < Y_min),
                           status_reason="no quality-certified candidate (Omega_F empty)")
            # base-error requirement: the largest e0 for which ANY allowed
            # candidate meets F_min at this length and launch power
            e0m = e0_max_for_certification(eta_s, nu_s, sig_s, q_star)
            if np.isfinite(e0m).any():
                j = int(np.nanargmax(e0m))
                row["e0_max_for_certification"] = float(e0m[j])
                row["e0_max_wavelength_nm"] = float(lam_allowed[j])
            else:
                row["e0_max_for_certification"] = np.nan
                row["e0_max_wavelength_nm"] = np.nan
            rows.append(row)
        print(f"  L = {L:4.0f} km done ({time.time()-t0:.0f} s)", flush=True)
    man.add_output(io.write_csv(os.path.join(d, "operating_map.csv"), rows))

    # --------------------------------------- 3x3 readable summary with exact Fe
    keys_id = pr.key_set("identity")
    for L in cfg.get("summary_lengths_km", [5.0, 10.0, 15.0]):
        sc = cf.build_scenario(cfg, length_km=float(L))
        comp = components_at_length(sc, f_allowed)
        for Pd in cfg.get("summary_powers_dbm", [-8.0, -5.0, -2.0]):
            k = float(dbm_to_w(Pd) / P_ref)
            eta_s, nu_s, sig_s, *_ = rescale(comp, k)
            pt = reduce_point(eta_s, nu_s, sig_s, e0)
            mF = (pt["s"] > 0) & (pt["C"] >= F_min)
            base = dict(length_km=float(L), launch_dbm_per_channel=float(Pd),
                        quality_domain_nonempty=bool(mF.any()))
            if mF.any():
                i = int(np.nonzero(mF)[0][np.argmax(pt["Y_lb"][mF])])
            else:                      # report the best available candidate honestly
                i = int(np.argmax(pt["Y_lb"]))
            p1 = pr.channel_point(eta_s[i], nu_s[i], e0, pt["z"][i])
            fe, _ = pr.exact_fidelity(p1, keys_id, "pauli_enumeration")
            mW = mF & (pt["Y_lb"] >= Y_min)
            base.update(lambda_star_nm=float(lam_allowed[i]),
                        P_acc_at_star=float(pt["P_acc"][i]),
                        C_bound_at_star=float(pt["C"][i]),
                        Y_lb_at_star=float(pt["Y_lb"][i]),
                        Y_ub_at_star=float(pt["Y_ub"][i]),
                        Fe_exact_at_star=fe, Y_exact_at_star=float(pt["P_acc"][i]) * fe,
                        gap_Fe_minus_C=fe - float(pt["C"][i]),
                        W_q_nonempty=bool(mW.any()), n_W_q=int(mW.sum()),
                        W_q_intervals_or_channel_count=len(
                            opt.intervals_from_mask(lam_allowed, mW, allowed_indices)),
                        N_epsilon_intervals_or_channel_count=len(
                            opt.intervals_from_mask(
                                lam_allowed, mF & (pt["Y_lb"] >= (1 - eps) * pt["Y_lb"][i]),
                                allowed_indices)),
                        status_reason=("certified" if mW.any() else
                                       "quality certified; yield not certified" if mF.any()
                                       else "NOT certified: Omega_F empty; the row reports "
                                            "the best available candidate, not a certified one"))
            summary.append(base)
    man.add_output(io.write_csv(os.path.join(d, "operating_points_3x3.csv"), summary))

    nc = sum(1 for r in rows if r["W_q_nonempty"])
    nq = sum(1 for r in rows if r["quality_domain_nonempty"])
    man.note(f"{nq}/{len(rows)} map points have a nonempty Omega_F; "
             f"{nc}/{len(rows)} certify both F_min and Y_min.")
    man.save(os.path.join(d, "run_manifest_map.json"))
    print(f"\noperating map: {len(rows)} points, Omega_F nonempty at {nq}, "
          f"both targets certified at {nc}  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
