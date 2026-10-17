#!/usr/bin/env python
"""Stage F: ablations, parameter/model sensitivity, and robust selection
(Methods: ablations, sensitivity and robust selection)."""
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

N, TT = 7, 1

# Ratio of the two published Raman levels at the Qi et al. separation: the
# near-carrier measurement of da Silva et al. (2014) over the Qi et al. value.
import raman as _rm
DASILVA_LEVEL_RATIO = float(_rm.dasilva_rho(abs(_rm.QI_DELTA_F_HZ))
                            / _rm.calibrated_rho(_rm.QI_DELTA_F_HZ))


def evaluate_set(cfg, f_allowed, length_km, launch_dbm, overrides=None,
                 atten_source=None):
    sc = cf.build_scenario(cfg, length_km=length_km, launch_dbm=launch_dbm,
                           overrides=overrides)
    if atten_source:
        sc.atten = cf.build_attenuation(cfg, source=atten_source)
        sc._fwm_cache = {}
    e0 = sc.e0_per_pass
    n = len(f_allowed)
    eta = np.zeros((n, 3)); nu = np.zeros((n, 3)); sig = np.zeros((n, 3))
    for i, f in enumerate(f_allowed):
        r = ol.evaluate_candidate(sc, f, want_components=False)
        eta[i] = r["eta"]; nu[i] = r["nu"]; sig[i] = r["sigma2_phi"]
    T, nuf, _ = oc.compose_passes(eta, nu)
    res = oc.accepted_optical(T, nuf)
    z = lp.z_from_sigma2(sig)
    b = np.stack([lp.traversal_weights(e0, z[..., m])[..., 0] for m in range(3)], axis=-1)
    q = lp.affected_probability(res.wI, b)
    C = rl.correctable_weight(q, N, TT)
    s = res.s
    return dict(T=T, nu=nuf, s=s, r=res.r, q=q, C=C, P_acc=s ** N,
                Y_lb=s ** N * C, Y_ub=s ** N, z=z, eta=eta, nu_pass=nu, e0=e0)


ABLATIONS = {
    "full_model": {},
    "no_raman": {"raman_scale": 0.0},
    "no_fwm": {"enable_fwm": False},
    "no_leakage": {"enable_leakage": False},
    "no_residual_xpm": {"enable_xpm": False},
    "no_additive_background": {"raman_scale": 0.0, "enable_fwm": False,
                               "enable_leakage": False},
}

SCENARIOS = [
    ("nominal", {}, None, "reference"),
    ("raman_scale_0.5", {"raman_scale": 0.5}, None, "parameter sensitivity: "
     "unsupported stress factor, not a measured uncertainty limit"),
    ("raman_scale_2.0", {"raman_scale": 2.0}, None, "parameter sensitivity: "
     "unsupported stress factor, not a measured uncertainty limit.  The "
     "polarization convention is NOT an additional factor of two: it is resolved "
     "in favour of the polarization-summed reading used in the baseline"),
    ("raman_shape_standin_multigaussian", {"raman_model": "multi_gaussian_v1"},
     None, "model sensitivity: earlier stand-in lineshape (not a fit to measured "
     "data), normalized at the same measurement point"),
    ("raman_shape_standin_oscillator", {"raman_model": "damped_oscillator_v1"},
     None, "model sensitivity: earlier stand-in damped-oscillator lineshape, "
     "normalized at the same measurement point"),
    ("raman_level_dasilva2014", {"raman_scale": DASILVA_LEVEL_RATIO}, None,
     "literature sensitivity: absolute level implied by the near-carrier "
     "measurement of da Silva et al. (2014), which is lower than the Qi et al. "
     "calibration at the same 0.6 THz separation"),
    # NOTE: a "per-polarization convention" scenario was removed.  The convention
    # is RESOLVED from the source (Qi et al. Eqs. 5, 6 and 15 apply the factor 1/2
    # for single-polarization selection only at the final step), so the
    # polarization-summed reading used here is the correct one and a factor-two
    # doubling is already covered by raman_scale_2.0 as an unsupported stress.
    ("attenuation_analytic_model", {}, "analytic_three_term",
     "model sensitivity: analytic three-term silica curve instead of the "
     "interpolated datasheet-maximum table"),
    ("base_error_halved", {"e0_per_pass": 0.001}, None, "parameter sensitivity"),
    ("base_error_doubled", {"e0_per_pass": 0.004}, None, "parameter sensitivity"),
    ("xpm_kappa_0.25", {"xpm_kappa": 0.25}, None, "parameter sensitivity: weaker "
     "differential H-V coupling"),
    ("leak_isolation_minus_10dB", {"leak_isolation_remote_db": 100.0,
                                   "leak_isolation_local_db": 80.0}, None,
     "parameter sensitivity: worse receiver/path isolation"),
    ("tx_floor_x10", {"tx_broadband_floor_per_hz": 2.5e-13}, None,
     "parameter sensitivity: higher transmitter broadband emission"),
    ("fwm_profile_1GHz", {"fwm_profile_rms_hz": 1e9}, None,
     "model sensitivity: PRESCRIBED 1 GHz rms spectral profile on every "
     "generated FWM component at fixed product power; not a modulated-FWM "
     "simulation"),
    ("fwm_profile_5GHz", {"fwm_profile_rms_hz": 5e9}, None,
     "model sensitivity: prescribed 5 GHz rms FWM profile"),
    ("fwm_profile_25GHz", {"fwm_profile_rms_hz": 25e9}, None,
     "model sensitivity: prescribed 25 GHz rms FWM profile, matching the "
     "declared classical modulation width"),
    ("receiver_power_average", {"receiver_mode_convention": "power_average"},
     None, "receiver-model sensitivity: the legacy power average "
     "int |psi|^2 T_q df instead of the fixed-mode overlap "
     "|int |psi|^2 sqrt(T_q) df|^2 used in the main results"),
]


def summarize(pt, lam, F_min, Y_min, eps, label, extra=None, grid_indices=None):
    mF = (pt["s"] > 0) & (pt["C"] >= F_min)
    out = dict(label=label, Omega_F_nonempty=bool(mF.any()), n_Omega_F=int(mF.sum()))
    # M_theta = max over the allowed domain (s > 0) of C_7,1(q).  Omega_F is
    # empty exactly when M_theta < F_min, so this is the quantity that must be
    # plotted against the empty/non-empty classification.
    ms = pt["s"] > 0
    if ms.any():
        j = int(np.nonzero(ms)[0][np.argmax(pt["C"][ms])])
        out.update(max_C_over_Omega_q=float(pt["C"][j]),
                   lambda_at_max_C_nm=float(lam[j]))
    else:
        out.update(max_C_over_Omega_q=np.nan, lambda_at_max_C_nm=np.nan)
    if mF.any():
        i = int(np.nonzero(mF)[0][np.argmax(pt["Y_lb"][mF])])
        mW = mF & (pt["Y_lb"] >= Y_min)
        out.update(lambda_nm=float(lam[i]), Y_lb=float(pt["Y_lb"][i]),
                   C_bound=float(pt["C"][i]), P_acc=float(pt["P_acc"][i]),
                   Y_ub=float(pt["Y_ub"][i]), q=float(pt["q"][i]),
                   target_status="certified" if mW.any() else
                   "quality certified; yield not certified",
                   W_q_intervals=len(opt.intervals_from_mask(lam, mW, grid_indices)),
                   N_epsilon_intervals=len(opt.intervals_from_mask(
                       lam, mF & (pt["Y_lb"] >= (1 - eps) * pt["Y_lb"][i]),
                       grid_indices)))
    else:
        i = int(np.argmax(pt["Y_lb"]))
        out.update(lambda_nm=np.nan, Y_lb=np.nan, C_bound=np.nan, P_acc=np.nan,
                   Y_ub=float(pt["Y_ub"][i]), q=float(pt["q"][i]),
                   target_status="NOT certified: Omega_F empty",
                   W_q_intervals=0, N_epsilon_intervals=0,
                   best_uncertified_lambda_nm=float(lam[i]),
                   best_uncertified_C_bound=float(pt["C"][i]),
                   best_uncertified_Y_lb=float(pt["Y_lb"][i]))
    if extra:
        out.update(extra)
    return out


def main():
    t0 = time.time()
    cfg = cf.load("illustrative_baseline.yaml")
    run_id = os.environ.get("QCFC_RUN_ID", cfg["scenario_id"])
    d = io.run_dir(run_id)
    man = io.Manifest(run_id, "F sensitivity and ablations", cfg)
    tg = cfg["targets"]
    F_min, Y_min, eps = tg["F_min"], tg["Y_min"], tg["epsilon"]

    dep = cf.build_deployment(cfg)
    grid = cf.build_grid(cfg, dep)
    _, _, _, allowed, _ = cf.build_masks(cfg, grid, dep)
    f_allowed = grid[allowed]
    allowed_indices = np.flatnonzero(allowed)
    lam = 299792458.0 / f_allowed * 1e9

    anchors = [("interior_2km_-15dBm", 2.0, -15.0),
               ("interior_3km_-13dBm", 3.0, -13.0),
               ("original_comparison_5km_-10dBm", 5.0, -10.0),
               ("legacy_comparison_10km_-5dBm", 10.0, -5.0)]

    # -------------------------------------------------------------- ablations
    abl_rows = []
    for aname, L, P in anchors:
        full = evaluate_set(cfg, f_allowed, L, P)
        mF = (full["s"] > 0) & (full["C"] >= F_min)
        i_full = (int(np.nonzero(mF)[0][np.argmax(full["Y_lb"][mF])]) if mF.any()
                  else int(np.argmax(full["Y_lb"])))
        for abname, ov in ABLATIONS.items():
            pt = evaluate_set(cfg, f_allowed, L, P, overrides=ov)
            # (1) evaluated at the FULL-MODEL selected wavelength
            abl_rows.append(dict(
                anchor=aname, length_km=L, launch_dbm=P,
                scenario_or_removed_impairment=abname,
                evaluation_mode="fixed at the full-model selected wavelength",
                lambda_nm=float(lam[i_full]), s=float(pt["s"][i_full]),
                q=float(pt["q"][i_full]), C_bound=float(pt["C"][i_full]),
                P_acc=float(pt["P_acc"][i_full]), Y_lb=float(pt["Y_lb"][i_full]),
                delta_lambda_from_nominal=0.0,
                relative_yield_change_if_defined=(
                    float(pt["Y_lb"][i_full] / full["Y_lb"][i_full] - 1.0)
                    if full["Y_lb"][i_full] > 0 else np.nan),
                target_status=("certified" if (pt["C"][i_full] >= F_min and
                                               pt["Y_lb"][i_full] >= Y_min)
                               else "not certified at this wavelength"),
                uncertainty_or_sensitivity_interpretation=(
                    "ablation: the named additive source is removed and nu, s, r, q "
                    "and all bounds are recomputed; no yield contribution is subtracted")))
            # (2) re-optimized after ablation
            s2 = summarize(pt, lam, F_min, Y_min, eps, abname,
                           grid_indices=allowed_indices)
            abl_rows.append(dict(
                anchor=aname, length_km=L, launch_dbm=P,
                scenario_or_removed_impairment=abname,
                evaluation_mode="re-optimized after ablation",
                lambda_nm=s2.get("lambda_nm", s2.get("best_uncertified_lambda_nm")),
                s=np.nan, q=s2["q"],
                C_bound=s2.get("C_bound", s2.get("best_uncertified_C_bound")),
                P_acc=s2.get("P_acc", np.nan),
                Y_lb=s2.get("Y_lb", s2.get("best_uncertified_Y_lb")),
                delta_lambda_from_nominal=(
                    (s2.get("lambda_nm", s2.get("best_uncertified_lambda_nm", np.nan))
                     - float(lam[i_full]))),
                relative_yield_change_if_defined=np.nan,
                target_status=s2["target_status"],
                uncertainty_or_sensitivity_interpretation=(
                    "removing background can LOWER s by removing replacement photons "
                    "even while improving conditional fidelity")))
    man.add_output(io.write_csv(os.path.join(d, "ablations.csv"), abl_rows))

    # ------------------------------------------------------------- scenarios
    sc_rows, sets = [], {}
    for aname, L, P in anchors:
        for sname, ov, asrc, interp in SCENARIOS:
            pt = evaluate_set(cfg, f_allowed, L, P, overrides=ov, atten_source=asrc)
            sets[(aname, sname)] = pt
            r = summarize(pt, lam, F_min, Y_min, eps, sname,
                          extra=dict(anchor=aname, length_km=L, launch_dbm=P,
                                     uncertainty_or_sensitivity_interpretation=interp),
                          grid_indices=allowed_indices)
            sc_rows.append(r)
    man.add_output(io.write_csv(os.path.join(d, "scenario_sensitivity.csv"), sc_rows))

    # --------------------------------------------------- robust selection
    rob_rows = []
    supported = [s for s in SCENARIOS if s[0] != "nominal"]
    for aname, L, P in anchors:
        names = ["nominal"] + [s[0] for s in supported]
        Fmasks = [(sets[(aname, n)]["s"] > 0) & (sets[(aname, n)]["C"] >= F_min)
                  for n in names]
        rob_mask = np.logical_and.reduce(Fmasks)
        worst = np.min(np.stack([sets[(aname, n)]["Y_lb"] for n in names]), axis=0)
        nom = sets[(aname, "nominal")]
        mF_nom = Fmasks[0]
        i_nom = (int(np.nonzero(mF_nom)[0][np.argmax(nom["Y_lb"][mF_nom])])
                 if mF_nom.any() else int(np.argmax(nom["Y_lb"])))
        if rob_mask.any():
            i_rob = int(np.nonzero(rob_mask)[0][np.argmax(worst[rob_mask])])
        else:
            i_rob = None
        # (a) robustness of the ONE nominal wavelength
        for n in names:
            pt = sets[(aname, n)]
            rob_rows.append(dict(
                anchor=aname, mode="fixed nominal wavelength", scenario=n,
                lambda_nm=float(lam[i_nom]), C_bound=float(pt["C"][i_nom]),
                Y_lb=float(pt["Y_lb"][i_nom]), P_acc=float(pt["P_acc"][i_nom]),
                certified=bool(pt["C"][i_nom] >= F_min and pt["Y_lb"][i_nom] >= Y_min),
                note="the nominal optimum evaluated under each scenario"))
        # (b) performance if the wavelength is retuned per scenario
        for n in names:
            pt = sets[(aname, n)]
            mF = (pt["s"] > 0) & (pt["C"] >= F_min)
            if mF.any():
                i = int(np.nonzero(mF)[0][np.argmax(pt["Y_lb"][mF])])
                rob_rows.append(dict(anchor=aname, mode="retuned per scenario",
                                     scenario=n, lambda_nm=float(lam[i]),
                                     C_bound=float(pt["C"][i]), Y_lb=float(pt["Y_lb"][i]),
                                     P_acc=float(pt["P_acc"][i]),
                                     certified=bool(pt["Y_lb"][i] >= Y_min),
                                     note="scenario-specific re-optimization"))
            else:
                rob_rows.append(dict(anchor=aname, mode="retuned per scenario",
                                     scenario=n, lambda_nm=np.nan, C_bound=np.nan,
                                     Y_lb=np.nan, P_acc=np.nan, certified=False,
                                     note="Omega_F empty under this scenario"))
        # (c) robust selection over the enumerated scenario set
        rob_rows.append(dict(
            anchor=aname, mode="robust selection over the enumerated scenario set",
            scenario="intersection of Omega_F; max of the worst-case Y_lb",
            lambda_nm=(float(lam[i_rob]) if i_rob is not None else np.nan),
            C_bound=(float(min(sets[(aname, n)]["C"][i_rob] for n in names))
                     if i_rob is not None else np.nan),
            Y_lb=(float(worst[i_rob]) if i_rob is not None else np.nan),
            P_acc=np.nan,
            certified=bool(i_rob is not None and worst[i_rob] >= Y_min),
            note=("certifies only the enumerated scenario set under its models; it "
                  "does not certify a continuous uncertainty region")))
    man.add_output(io.write_csv(os.path.join(d, "robustness_summary.csv"), rob_rows))

    # --------------------------- ensemble averaging over a shared uncertain state
    ens_rows = []
    keys_id = pr.key_set("identity")
    for aname, L, P in anchors:
        names = ["nominal"] + [s[0] for s in supported]
        i_nom = None
        nom = sets[(aname, "nominal")]
        mF_nom = (nom["s"] > 0) & (nom["C"] >= F_min)
        i_nom = (int(np.nonzero(mF_nom)[0][np.argmax(nom["Y_lb"][mF_nom])])
                 if mF_nom.any() else int(np.argmax(nom["Y_lb"])))
        Pacc, Yex, Fecond = [], [], []
        for n in names:
            pt = sets[(aname, n)]
            p1 = pr.channel_point(pt["eta"][i_nom], pt["nu_pass"][i_nom], pt["e0"],
                                  pt["z"][i_nom])
            fe, _ = pr.exact_fidelity(p1, keys_id, "pauli_enumeration")
            Pacc.append(p1["P_acc"]); Yex.append(p1["P_acc"] * fe); Fecond.append(fe)
        Pbar, Ybar = float(np.mean(Pacc)), float(np.mean(Yex))
        ens_rows.append(dict(
            anchor=aname, lambda_nm=float(lam[i_nom]), n_scenarios=len(names),
            mean_P_acc=Pbar, mean_Y_exact=Ybar,
            acceptance_conditioned_ensemble_fidelity=(Ybar / Pbar if Pbar > 0 else np.nan),
            unweighted_mean_conditional_fidelity=float(np.mean(Fecond)),
            note=("nonlinear metrics are computed per sample and then averaged; the "
                  "acceptance-conditioned ensemble fidelity is Ybar/Pbar, which is "
                  "NOT the unweighted mean of the conditional fidelities")))
    man.add_output(io.write_csv(os.path.join(d, "ensemble_averaging.csv"), ens_rows))

    # ------------------------------------------ selector stability across the map
    # The quality constraint is active only near the certification boundary.  This
    # sweep records whether the selector ordering survives away from that boundary.
    stab = []
    for L, P in [(5.0, -10.0), (5.0, -13.0), (4.0, -12.0), (3.0, -13.0),
                 (3.0, -10.0), (2.0, -15.0), (2.0, -8.0), (1.0, -5.0)]:
        pt = evaluate_set(cfg, f_allowed, L, P)
        mF = (pt["s"] > 0) & (pt["C"] >= F_min)
        if not mF.any():
            stab.append(dict(length_km=L, launch_dbm=P, Omega_F_nonempty=False,
                             n_Omega_F=0, quality_constraint_active=np.nan,
                             note="Omega_F empty"))
            continue
        iY = int(np.nonzero(mF)[0][np.argmax(pt["Y_lb"][mF])])
        iT = int(np.argmax(pt["T"])); iq = int(np.argmin(pt["q"]))
        nN = int((mF & (pt["Y_lb"] >= (1 - eps) * pt["Y_lb"][iY])).sum())
        stab.append(dict(
            length_km=L, launch_dbm=P, Omega_F_nonempty=True,
            n_allowed=int(len(f_allowed)), n_Omega_F=int(mF.sum()),
            quality_constraint_active=bool(mF.sum() < len(f_allowed)),
            C_min_over_Omega_q=float(pt["C"].min()),
            C_max_over_Omega_q=float(pt["C"].max()),
            lambda_maxYlb_nm=float(lam[iY]),
            margin_C_minus_Fmin_at_star=float(pt["C"][iY] - F_min),
            lambda_maxT_nm=float(lam[iT]), C_at_maxT=float(pt["C"][iT]),
            maxT_selector_certifies=bool(pt["C"][iT] >= F_min),
            lambda_minq_nm=float(lam[iq]),
            minq_relative_Ylb_loss=float(1 - pt["Y_lb"][iq] / pt["Y_lb"][iY]),
            n_N_epsilon=nN,
            N_epsilon_fraction_of_Omega_F=float(nN / mF.sum()),
            note=("quality constraint ACTIVE: some allowed candidates fail F_min"
                  if mF.sum() < len(f_allowed) else
                  "quality constraint INACTIVE: Omega_F = Omega_q, every allowed "
                  "candidate certifies, so the loss-only selector certifies too")))
    man.add_output(io.write_csv(os.path.join(d, "selector_stability.csv"), stab))

    man.note("The quality constraint binds only near the certification boundary; "
             "the selector comparison must be read with that in mind.")
    man.note("Sensitivity envelopes here are NOT confidence intervals: no source "
             "supplied a defensible distribution for any optical input.")
    man.save(os.path.join(d, "run_manifest_sensitivity.json"))
    print(f"Stage F complete: {len(abl_rows)} ablation rows, {len(sc_rows)} scenario "
          f"rows, {len(rob_rows)} robustness rows ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
