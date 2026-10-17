#!/usr/bin/env python
"""Stage D: fixed-deployment wavelength scan, selection, certified regions, and
the convergence rerun (Methods: wavelength scan and selection)."""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import config as cf
import deployment as dpl
import optimization as opt
import protocol as pr
import reliability as rl
import runio as io
import mode_reference as mref
from units import C_LIGHT, nm_to_hz, hz_to_nm

SCAN_COLS = ["run_id", "scenario_id", "wavelength_nm", "frequency_hz",
             "hardware_allowed", "guard_allowed", "search_allowed", "exclusion_reason",
             "eta_1", "eta_2", "eta_3", "T",
             "nu_R_1", "nu_R_2", "nu_R_3", "nu_FWM_1", "nu_FWM_2", "nu_FWM_3",
             "nu_leak_1", "nu_leak_2", "nu_leak_3", "nu_1", "nu_2", "nu_3",
             "nu_R_final", "nu_FWM_final", "nu_leak_final", "nu_final",
             "sigma2_phi_1", "sigma2_phi_2", "sigma2_phi_3",
             "z_1", "z_2", "z_3", "e0_1", "e0_2", "e0_3",
             "c0", "c1", "s", "r", "wI", "b_1", "b_2", "b_3", "q",
             "C_bound", "P_acc", "Y_lb", "Y_ub", "log_Y_lb", "log_Y_ub",
             "Fe_exact", "Y_exact", "exact_decoder_evaluated", "private_key_set_id",
             "in_Omega_F", "in_W_q", "in_N_epsilon",
             "thermal_model_status", "calibration_status"]


def scan_scenario(cfg, run_id, step_ghz=None):
    dep = cf.build_deployment(cfg)
    sc = cf.build_scenario(cfg)
    grid = cf.build_grid(cfg, dep, step_ghz)
    hw, guard, search, allowed, reason = cf.build_masks(cfg, grid, dep)
    cols = opt.scan_wavelengths(sc, grid, e0_per_pass=cfg["link"]["e0_per_pass"])
    cols["hardware_allowed"] = hw; cols["guard_allowed"] = guard
    cols["search_allowed"] = search; cols["allowed"] = allowed
    cols["exclusion_reason"] = reason
    for j in range(3):
        cols[f"e0_{j+1}"] = np.full(len(grid), cfg["link"]["e0_per_pass"])
    return sc, dep, grid, cols, allowed


def main():
    t_start = time.time()
    cfg = cf.load(os.environ.get("QCFC_BASELINE_CONFIG", "illustrative_baseline.yaml"))
    # Predeclared operating point / margin overrides, written back into the
    # configuration so the exported provenance describes the executed run.
    if os.environ.get("QCFC_LENGTH_KM"):
        cfg["link"]["length_km_per_traversal"] = float(os.environ["QCFC_LENGTH_KM"])
    if os.environ.get("QCFC_LAUNCH_DBM"):
        cfg["deployment"]["classical_launch_dbm_per_channel"] = float(
            os.environ["QCFC_LAUNCH_DBM"])
    if os.environ.get("QCFC_MARGIN_GHZ"):
        cfg["search"]["tuning_margin_ghz"] = float(os.environ["QCFC_MARGIN_GHZ"])
    if os.environ.get("QCFC_FWM_PROFILE_GHZ"):
        cfg["optics"]["fwm_profile_rms_ghz"] = float(
            os.environ["QCFC_FWM_PROFILE_GHZ"])
    run_id = os.environ.get("QCFC_RUN_ID", cfg["scenario_id"])
    d = io.run_dir(run_id)
    man = io.Manifest(run_id, "D baseline wavelength scan", cfg)
    tg = cfg["targets"]
    F_min, Y_min, eps = tg["F_min"], tg["Y_min"], tg["epsilon"]

    print(f"[{run_id}] scanning ...", flush=True)
    sc, dep, grid, cols, allowed = scan_scenario(cfg, run_id)

    # ------------------------------------------------------- frozen deployment
    man.add_output(io.write_csv(os.path.join(d, "classical_channels.csv"),
                                dep.to_rows()))
    ledger = []
    for m in range(3):
        cl = dpl.ComponentLedger(sc.eta_tx, sc.eta_rx,
                                 note=f"legacy eta_insertion_per_pass="
                                      f"{cfg['link']['eta_insertion_per_pass']} split "
                                      f"symmetrically; contents never specified")
        for row in cl.rows(m, sc.length_m):
            row["direction"] = dep.quantum_directions[m]
            row["raman_direction"] = "co" if dep.copropagating(m) else "counter"
            row["leakage_path"] = "local" if dep.local_transmitter(m) else "remote"
            row["fwm_collected"] = bool(dep.quantum_directions[m] == dep.direction)
            ledger.append(row)
    man.add_output(io.write_csv(os.path.join(d, "component_paths.csv"), ledger))

    # ------------------------------------------------------------ feasible sets
    in_F = allowed & (cols["s"] > 0) & (cols["C_bound"] >= F_min)
    n_F = int(in_F.sum())
    if n_F:
        k_star = int(np.nonzero(in_F)[0][np.argmax(cols["Y_lb"][in_F])])
        Ylb_star = float(cols["Y_lb"][k_star])
    else:
        k_star, Ylb_star = None, np.nan
    in_W = in_F & (cols["Y_lb"] >= Y_min)
    in_N = in_F & (cols["Y_lb"] >= (1 - eps) * Ylb_star) if n_F else np.zeros_like(in_F)

    # --------------------------------------------- exact decoder at key points
    keys_id = pr.key_set("identity")
    keys_nc = pr.key_set("fixed_nonclifford", axis="y", theta_A=0.6283185307179586,
                         theta_B=1.0471975511965976)
    eval_idx = set()
    if k_star is not None:
        eval_idx.add(k_star)
    ref_idx, ref_status = opt.fixed_reference_index(cols, allowed,
                                                    tg["fixed_reference_nm"])
    near_idx = opt.nearest_allowed_index(cols, allowed, tg["fixed_reference_nm"])
    for _k in (ref_idx, near_idx):
        if _k is not None:
            eval_idx.add(_k)
    idx_allowed = np.nonzero(allowed)[0]
    # Exact identity-key fidelity is affordable for EVERY allowed candidate
    # (Pauli enumeration over all 4^7 net strings costs ~2 ms per candidate),
    # so the exact curve is computed over the whole allowed domain rather than
    # at a handful of points.  The non-Clifford density-matrix route stays at
    # the selected, comparator, dominant-noise and near-threshold candidates.
    eval_all = list(idx_allowed)
    if len(idx_allowed):
        eval_idx.add(int(idx_allowed[np.argmax(cols["T"][idx_allowed])]))
        eval_idx.add(int(idx_allowed[np.argmin(cols["q"][idx_allowed])]))
        eval_idx.add(int(idx_allowed[np.argmax(cols["Y_lb"][idx_allowed])]))
        eval_idx.add(int(idx_allowed[np.argmax(cols["C_bound"][idx_allowed])]))
        # local maxima of Y_lb among allowed candidates
        y = cols["Y_lb"][idx_allowed]
        for u in range(1, len(idx_allowed) - 1):
            if y[u] >= y[u - 1] and y[u] >= y[u + 1]:
                eval_idx.add(int(idx_allowed[u]))
        # near the certification boundary
        order = np.argsort(np.abs(cols["C_bound"][idx_allowed] - F_min))
        eval_idx.update(int(idx_allowed[j]) for j in order[:5])
    eval_idx = sorted(eval_idx)[:80]

    Fe = np.full(len(grid), np.nan); Ye = np.full(len(grid), np.nan)
    flag = np.zeros(len(grid), dtype=bool)
    keyid = np.array(["" for _ in grid], dtype=object)
    for k in eval_all:
        pt = pr.channel_point([cols[f"eta_{j+1}"][k] for j in range(3)],
                              [cols[f"nu_{j+1}"][k] for j in range(3)],
                              cfg["link"]["e0_per_pass"],
                              [cols[f"z_{j+1}"][k] for j in range(3)])
        fe_id, _ = pr.exact_fidelity(pt, keys_id, "pauli_enumeration")
        Fe[k] = fe_id; Ye[k] = pt["P_acc"] * fe_id; flag[k] = True
        keyid[k] = "identity"
    exact_rows = []
    for k in eval_idx:
        pt = pr.channel_point([cols[f"eta_{j+1}"][k] for j in range(3)],
                              [cols[f"nu_{j+1}"][k] for j in range(3)],
                              cfg["link"]["e0_per_pass"],
                              [cols[f"z_{j+1}"][k] for j in range(3)])
        fe_id, _ = pr.exact_fidelity(pt, keys_id, "pauli_enumeration")
        fe_dm, _ = pr.exact_fidelity(pt, keys_id, "density_matrix")
        fe_nc, _ = pr.exact_fidelity(pt, keys_nc, "density_matrix")
        Fe[k] = fe_id; Ye[k] = pt["P_acc"] * fe_id; flag[k] = True
        keyid[k] = "identity"
        exact_rows.append(dict(
            wavelength_nm=cols["wavelength_nm"][k], q=pt["q"], s=pt["s"],
            C_bound=pt["C_bound"], P_acc=pt["P_acc"], Y_lb=pt["Y_lb"], Y_ub=pt["Y_ub"],
            Fe_identity_enumeration=fe_id, Fe_identity_density_matrix=fe_dm,
            Fe_fixed_nonclifford=fe_nc,
            enumeration_vs_density_matrix=abs(fe_id - fe_dm),
            Y_exact_identity=pt["P_acc"] * fe_id,
            Y_exact_nonclifford=pt["P_acc"] * fe_nc,
            gap_identity=fe_id - pt["C_bound"], gap_nonclifford=fe_nc - pt["C_bound"],
            bound_satisfied=bool(min(fe_id, fe_nc) >= pt["C_bound"] - 1e-12),
            uncoded_n1_Fe=pr.uncoded_fidelity(pt)))
    man.add_output(io.write_csv(os.path.join(d, "exact_decoder_points.csv"), exact_rows))

    cols["Fe_exact"] = Fe; cols["Y_exact"] = Ye
    cols["exact_decoder_evaluated"] = flag; cols["private_key_set_id"] = keyid
    cols["in_Omega_F"] = in_F; cols["in_W_q"] = in_W; cols["in_N_epsilon"] = in_N
    cols["run_id"] = np.array([run_id] * len(grid), dtype=object)
    cols["scenario_id"] = np.array([cfg["scenario_id"]] * len(grid), dtype=object)
    cols["thermal_model_status"] = np.array(
        [cfg["thermal_model_status"]] * len(grid), dtype=object)
    cols["calibration_status"] = np.array(
        [cfg["calibration_status"]] * len(grid), dtype=object)
    # nu contributions must sum to nu_final
    resid = float(np.max(np.abs(cols["nu_R_final"] + cols["nu_FWM_final"] +
                                cols["nu_leak_final"] - cols["nu_final"])))
    man.tolerance("nu_contribution_sum_residual", resid)
    man.add_output(io.write_csv(os.path.join(d, "wavelength_scan.csv"),
                                io.columns_to_rows(cols, SCAN_COLS), SCAN_COLS))

    # ---------------------------------------------------------- certified regions
    reg_rows = []
    for label, mask in (("Omega_q", allowed), ("Omega_F", in_F),
                        ("W_q", in_W), ("N_epsilon", in_N)):
        iv = opt.intervals_from_mask(cols["wavelength_nm"], mask)
        if not iv:
            reg_rows.append(dict(region=label, n_intervals=0, interval_index="",
                                 lambda_lo_nm="", lambda_hi_nm="", width_nm="",
                                 n_grid_points=0,
                                 status="EMPTY under the stated model and targets"))
        for i, (lo, hi) in enumerate(iv):
            npts = int(np.count_nonzero(mask & (cols["wavelength_nm"] >= lo) &
                                        (cols["wavelength_nm"] <= hi)))
            reg_rows.append(dict(region=label, n_intervals=len(iv), interval_index=i,
                                 lambda_lo_nm=lo, lambda_hi_nm=hi, width_nm=hi - lo,
                                 n_grid_points=npts,
                                 status="grid-endpoint span; isolated points have zero width"))
    man.add_output(io.write_csv(os.path.join(d, "certified_regions.csv"), reg_rows))

    # --------------------------------------------------------------- selectors
    sel = opt.selectors(cols, allowed, in_F)
    sel_rows = []

    def sel_row(rule, k, dom):
        if k is None:
            return dict(selection_rule=rule, lambda_selected_nm="",
                        searched_over=dom, certificate_status="undefined: empty set")
        return dict(selection_rule=rule, lambda_selected_nm=cols["wavelength_nm"][k],
                    searched_over=dom, T=cols["T"][k], s=cols["s"][k],
                    P_acc=cols["P_acc"][k], q=cols["q"][k],
                    C_bound=cols["C_bound"][k], Y_lb=cols["Y_lb"][k],
                    Y_ub=cols["Y_ub"][k],
                    Fe_exact=(cols["Fe_exact"][k] if flag[k] else "not evaluated"),
                    Y_exact=(cols["Y_exact"][k] if flag[k] else "not evaluated"),
                    in_W_q=bool(in_W[k]), in_N_epsilon=bool(in_N[k]),
                    certificate_status=("certified for F_min and Y_min" if in_W[k] else
                                        "quality certified, yield not certified" if in_F[k] else
                                        "NOT certified by the lower bound"))
    for key, res in sel.items():
        sel_rows.append(sel_row(res.rule if res else key,
                                res.index if res else None,
                                res.searched_over if res else "-"))
        sel_rows[-1]["selection_rule"] = key
    sel_rows.append(sel_row("fixed_reference_wavelength", ref_idx, "Omega_q"))
    sel_rows[-1]["selection_rule"] = (
        f"fixed reference {tg['fixed_reference_nm']} nm (exact)")
    sel_rows[-1]["certificate_status"] = (
        sel_rows[-1]["certificate_status"] if ref_idx is not None else ref_status)
    sel_rows.append(sel_row("nearest_allowed_to_reference", near_idx, "Omega_q"))
    sel_rows[-1]["selection_rule"] = (
        f"nearest allowed candidate to {tg['fixed_reference_nm']} nm "
        f"(a DISTINCT rule)")
    man.add_output(io.write_csv(
        os.path.join(d, "selection_summary.csv"), sel_rows,
        ["selection_rule", "lambda_selected_nm", "searched_over", "T", "s", "P_acc",
         "q", "C_bound", "Y_lb", "Y_ub", "Fe_exact", "Y_exact", "in_W_q",
         "in_N_epsilon", "certificate_status"]))

    # ------------------------------------------------- target sensitivity sweep
    tgt_rows = []
    for fm in tg["F_min_sensitivity"]:
        m_F = allowed & (cols["s"] > 0) & (cols["C_bound"] >= fm)
        if m_F.any():
            kk = int(np.nonzero(m_F)[0][np.argmax(cols["Y_lb"][m_F])])
            best = float(cols["Y_lb"][kk]); lam = float(cols["wavelength_nm"][kk])
        else:
            kk, best, lam = None, np.nan, np.nan
        for ym in tg["Y_min_sensitivity"]:
            m_W = m_F & (cols["Y_lb"] >= ym)
            tgt_rows.append(dict(F_min=fm, Y_min=ym,
                                 Omega_F_nonempty=bool(m_F.any()),
                                 n_Omega_F=int(m_F.sum()),
                                 lambda_star_nm=lam, max_Y_lb=best,
                                 W_q_nonempty=bool(m_W.any()), n_W_q=int(m_W.sum()),
                                 W_q_intervals=len(opt.intervals_from_mask(
                                     cols["wavelength_nm"], m_W)),
                                 max_Y_ub_over_Omega_q=float(
                                     np.max(cols["Y_ub"][allowed])) if allowed.any() else np.nan,
                                 yield_excluded_by_upper_bound=bool(
                                     allowed.any() and np.max(cols["Y_ub"][allowed]) < ym)))
    man.add_output(io.write_csv(os.path.join(d, "target_sensitivity.csv"), tgt_rows))

    # --------------------------------------------------------- convergence reruns
    # Grid refinement and quadrature refinement are varied SEPARATELY so that a
    # change can be attributed to one of them (Methods: convergence criteria).
    def _solve(cfg_x, step):
        _, _, g_x, c_x, a_x = scan_scenario(cfg_x, run_id, step_ghz=step)
        mF = a_x & (c_x["s"] > 0) & (c_x["C_bound"] >= F_min)
        if mF.any():
            kk = int(np.nonzero(mF)[0][np.argmax(c_x["Y_lb"][mF])])
            return (int(mF.sum()), float(c_x["wavelength_nm"][kk]),
                    float(c_x["Y_lb"][kk]), float(c_x["C_bound"][kk]))
        return (0, np.nan, np.nan, np.nan)

    base_step = cfg["search"]["grid_step_ghz"]
    gh0 = cfg["numerics"]["gauss_hermite_order"]
    gh1 = cfg["numerics"]["gauss_hermite_check_order"]
    conv = []
    print(f"[{run_id}] convergence: grid refinement at fixed quadrature ...",
          flush=True)
    for step in (base_step / 2.0, base_step / 4.0):
        cfg_g = cf.load(cfg["_path"])
        cfg_g["search"].update(cfg["search"])
        cfg_g["link"].update(cfg["link"])
        cfg_g["deployment"].update(cfg["deployment"])
        cfg_g["optics"].update(cfg["optics"])
        n_x, lam_x, y_x, c_x = _solve(cfg_g, step)
        conv.append(dict(refinement="wavelength grid", setting=f"{step:g} GHz step",
                         held_fixed=f"Gauss-Hermite order {gh0}",
                         n_Omega_F=n_x, lambda_star_nm=lam_x, max_Y_lb=y_x,
                         C_at_star=c_x,
                         relative_change_Y_lb=(abs(y_x - Ylb_star) / Ylb_star
                                               if np.isfinite(Ylb_star) and Ylb_star > 0
                                               and np.isfinite(y_x) else np.nan),
                         absolute_change_C=(abs(c_x - float(cols["C_bound"][k_star]))
                                            if k_star is not None and np.isfinite(c_x)
                                            else np.nan),
                         lambda_shift_nm=(abs(lam_x - float(cols["wavelength_nm"][k_star]))
                                          if k_star is not None and np.isfinite(lam_x)
                                          else np.nan)))
    print(f"[{run_id}] convergence: quadrature refinement at fixed grid ...",
          flush=True)
    cfg_q = cf.load(cfg["_path"])
    for sec in ("search", "link", "deployment", "optics"):
        cfg_q[sec].update(cfg[sec])
    cfg_q["numerics"]["gauss_hermite_order"] = gh1
    n_q, lam_q_, y_q, c_q = _solve(cfg_q, base_step)
    conv.append(dict(refinement="quadrature", setting=f"Gauss-Hermite order {gh1}",
                     held_fixed=f"{base_step:g} GHz grid step",
                     n_Omega_F=n_q, lambda_star_nm=lam_q_, max_Y_lb=y_q,
                     C_at_star=c_q,
                     relative_change_Y_lb=(abs(y_q - Ylb_star) / Ylb_star
                                           if np.isfinite(Ylb_star) and Ylb_star > 0
                                           and np.isfinite(y_q) else np.nan),
                     absolute_change_C=(abs(c_q - float(cols["C_bound"][k_star]))
                                        if k_star is not None and np.isfinite(c_q)
                                        else np.nan),
                     lambda_shift_nm=(abs(lam_q_ - float(cols["wavelength_nm"][k_star]))
                                      if k_star is not None and np.isfinite(lam_q_)
                                      else np.nan)))
    conv.append(dict(refinement="reference solve",
                     setting=f"{base_step:g} GHz grid, Gauss-Hermite order {gh0}",
                     held_fixed="-", n_Omega_F=n_F,
                     lambda_star_nm=(float(cols["wavelength_nm"][k_star])
                                     if k_star is not None else np.nan),
                     max_Y_lb=Ylb_star,
                     C_at_star=(float(cols["C_bound"][k_star])
                                if k_star is not None else np.nan),
                     relative_change_Y_lb=0.0, absolute_change_C=0.0,
                     lambda_shift_nm=0.0))
    conv.append(dict(refinement="nu contribution sum residual", setting="-",
                     held_fixed="-", n_Omega_F=n_F, lambda_star_nm=np.nan,
                     max_Y_lb=np.nan, C_at_star=np.nan,
                     relative_change_Y_lb=np.nan, absolute_change_C=resid,
                     lambda_shift_nm=np.nan))
    man.add_output(io.write_csv(os.path.join(d, "convergence.csv"), conv))

    # FWM product distribution over occupied and vacant grid positions.  Products
    # populate the WHOLE grid; the guard rule is what makes only the vacant
    # positions reachable by an allowed quantum candidate.
    prods = sc.fwm_products()
    fg = np.round(prods.freq_hz / 1e9).astype(int)
    occ = set(np.round(dep.freqs_hz / 1e9).astype(int).tolist())
    slots = set(range(int(min(occ)), int(max(occ)) + 1, 100))
    vac = slots - occ
    Pw = prods.power_incoherent_w
    m_occ = np.array([f in occ for f in fg])
    m_vac = np.array([f in vac for f in fg])
    m_out = ~(m_occ | m_vac)
    tot = float(Pw.sum()) or 1.0
    man.add_output(io.write_csv(os.path.join(d, "fwm_slot_split.csv"), [dict(
        n_product_lines=int(len(fg)),
        all_lines_on_the_100GHz_grid=bool(np.all(fg % 100 == 0)),
        n_occupied_slots=int(m_occ.sum()), n_vacant_in_band_slots=int(m_vac.sum()),
        n_out_of_band=int(m_out.sum()),
        power_occupied_w=float(Pw[m_occ].sum()),
        power_vacant_in_band_w=float(Pw[m_vac].sum()),
        power_out_of_band_w=float(Pw[m_out].sum()),
        power_fraction_occupied=float(Pw[m_occ].sum() / tot),
        power_fraction_vacant_in_band=float(Pw[m_vac].sum() / tot),
        vacant_in_band_slots_ghz=str(sorted(vac)),
        note="products land on occupied and vacant grid positions alike; the "
             "50 GHz guard rule excludes every occupied neighbourhood, so only "
             "vacant positions can coincide with an allowed quantum candidate")]))

    man.add_output(io.write_csv(os.path.join(d, "provenance.csv"),
                                io.provenance_rows(cfg, sc)))

    # Independent full-chain spectral reference for the declared receiver.
    ref_rows = []
    ridx = [k for k in (k_star, ref_idx, near_idx) if k is not None]
    idx_allowed2 = np.nonzero(allowed)[0]
    if len(idx_allowed2):
        ridx += [int(idx_allowed2[np.argmax(cols["C_bound"][idx_allowed2])]),
                 int(idx_allowed2[np.argmax(cols["Y_lb"][idx_allowed2])]),
                 int(idx_allowed2[len(idx_allowed2) // 2])]
    for k in sorted(set(ridx)):
        r = mref.full_chain_reference(
            sc, float(cols["frequency_hz"][k]),
            e0_per_pass=cfg["link"]["e0_per_pass"],
            sigma2_phi=[cols[f"sigma2_phi_{j+1}"][k] for j in range(3)])
        r["candidate_role"] = ("lambda_star" if k == k_star else
                               "fixed_reference" if k == ref_idx else
                               "nearest_allowed_to_reference" if k == near_idx
                               else "allowed")
        r.pop("nu_ref_per_traversal_contribution", None)
        ref_rows.append(r)
    if ref_rows:
        man.add_output(io.write_csv(os.path.join(d, "mode_reference.csv"), ref_rows))
    man.note(f"Omega_F {'nonempty' if n_F else 'EMPTY'} at F_min={F_min}: "
             f"{n_F} of {int(allowed.sum())} allowed candidates.")
    man.note("Optical inputs are illustrative; see provenance.csv. No measured "
             "spectrum, receiver characterization, or deployment record was supplied.")
    man.save(os.path.join(d, "run_manifest_baseline.json"))

    print(f"\n[{run_id}] {len(grid)} grid points, {int(allowed.sum())} allowed")
    print(f"  max C_bound over Omega_q = {np.max(cols['C_bound'][allowed]):.6f} "
          f"at {cols['wavelength_nm'][allowed][np.argmax(cols['C_bound'][allowed])]:.3f} nm")
    print(f"  F_min = {F_min}: Omega_F has {n_F} candidates")
    if n_F:
        print(f"  lambda* = {cols['wavelength_nm'][k_star]:.4f} nm, "
              f"Y_lb* = {Ylb_star:.4e}, C = {cols['C_bound'][k_star]:.6f}")
    print(f"  max Y_ub over Omega_q = {np.max(cols['Y_ub'][allowed]):.4e} "
          f"(Y_min = {Y_min})")
    print(f"  wall time {time.time()-t_start:.1f} s")
    return cols, allowed, in_F


if __name__ == "__main__":
    main()
