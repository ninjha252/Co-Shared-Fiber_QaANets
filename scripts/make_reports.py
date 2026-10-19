#!/usr/bin/env python
"""Reports built from stored results only (Methods: reporting):

  changes.md                  every correction, its basis, and what it changed
  validation_report.md        verification / assumptions / sensitivity / open
  captions.md                 standalone figure captions
  results_evidence_outline.md one paragraph per main figure
  run_metadata.json           configs, hashes, seeds, settings, output paths

Nothing here recomputes physics; every number is read from a CSV or manifest.
"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import runio as io
import config as cf

RES = os.path.join(ROOT, "results")
PTS = [("op_2km_m15dBm", 2.0, -15.0, "interior example"),
       ("op_3km_m13dBm", 3.0, -13.0, "interior example"),
       ("op_5km_m10dBm", 5.0, -10.0, "original comparison point"),
       ("op_10km_m5dBm", 10.0, -5.0, "legacy comparison point")]
SUMMARY_POINTS = PTS[:3] + [
    ("op_6km_m10dBm", 6.0, -10.0, "active fidelity constraint comparison")
] + PTS[3:]
ZERO = {"op_2km_m15dBm": "op_2km_m15dBm_margin0",
        "op_3km_m13dBm": "op_3km_m13dBm_margin0",
        "op_5km_m10dBm": "op_5km_m10dBm_margin0",
        "op_10km_m5dBm": "op_10km_m5dBm_margin0"}
MAIN = "op_5km_m10dBm"
SCN_ANCHOR = "original_comparison_5km_-10dBm"


def rd(run_id, fname):
    """Read a stored result table.

    Large per-candidate scans are distributed gzip-compressed to keep the
    repository small; pandas reads either form transparently.
    """
    p = os.path.join(RES, run_id, fname)
    if os.path.exists(p):
        return pd.read_csv(p)
    if os.path.exists(p + ".gz"):
        return pd.read_csv(p + ".gz")
    return None


def g1(df, col, default=np.nan):
    if df is None or col not in df or not len(df):
        return default
    v = df[col].iloc[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def fmt(x, n=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "undefined" if not np.isfinite(v) else f"{v:.{n}g}"


def point_summary():
    """One row per manuscript operating-point comparison, from stored scans."""
    rows = []
    for rid, L, P, role in SUMMARY_POINTS:
        sel = rd(rid, "selection_summary.csv")
        scan = rd(rid, "wavelength_scan.csv")
        if sel is None or scan is None:
            continue
        al = scan["exclusion_reason"] == "allowed"
        star = sel[sel["selection_rule"] == "max_Ylb_subject_to_Fmin"]
        z = rd(ZERO[rid], "selection_summary.csv") if rid in ZERO else None
        zstar = (z[z["selection_rule"] == "max_Ylb_subject_to_Fmin"]
                 if z is not None else None)
        r = dict(run_id=rid, length_km=L, launch_dbm=P, role=role,
                 n_allowed=int(al.sum()),
                 n_Omega_F=int((scan["in_Omega_F"] == True).sum()),
                 n_W_q=int((scan["in_W_q"] == True).sum()),
                 max_C=float(scan.loc[al, "C_bound"].max()),
                 lam=g1(star, "lambda_selected_nm"), s=g1(star, "s"),
                 q=g1(star, "q"), C=g1(star, "C_bound"),
                 Fe=g1(star, "Fe_exact"), Ylb=g1(star, "Y_lb"),
                 Yex=g1(star, "Y_exact"), Yub=g1(star, "Y_ub"),
                 status=g1(star, "certificate_status"),
                 lam0=(g1(zstar, "lambda_selected_nm") if zstar is not None
                       else np.nan),
                 Ylb0=(g1(zstar, "Y_lb") if zstar is not None else np.nan))
        for rule in ("max_T_over_Omega_q", "min_q_over_Omega_F"):
            rr = sel[sel["selection_rule"] == rule]
            r[f"lam_{rule}"] = g1(rr, "lambda_selected_nm")
            r[f"C_{rule}"] = g1(rr, "C_bound")
            r[f"Ylb_{rule}"] = g1(rr, "Y_lb")
            r[f"status_{rule}"] = g1(rr, "certificate_status")
        ref = sel[sel["selection_rule"].astype(str).str.startswith("fixed reference")]
        r["fixed_status"] = g1(ref, "certificate_status")
        near = sel[sel["selection_rule"].astype(str).str.startswith("nearest allowed")]
        r["near_lam"] = g1(near, "lambda_selected_nm")
        r["near_C"] = g1(near, "C_bound")
        r["near_status"] = g1(near, "certificate_status")
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    cfg = cf.load("illustrative_baseline.yaml")
    F_min = cfg["targets"]["F_min"]; Y_min = cfg["targets"]["Y_min"]
    guard = cfg["search"]["guard_band_ghz"]
    margin = cfg["search"].get("tuning_margin_ghz", 0.0)

    # ------------------------------------------------------------ validation
    s1o = rd("validation", "validation_table_s2_optics.csv")
    s1d = rd("validation", "validation_table_s2_decoder.csv")
    s1 = pd.concat([x for x in (s1o, s1d) if x is not None], ignore_index=True)
    if "row_kind" not in s1:
        s1["row_kind"] = "assert"
    n_assert = int((s1["row_kind"] == "assert").sum())
    n_diag = int((s1["row_kind"] != "assert").sum())
    n_fail = int(s1["pass_fail_and_reason"].astype(str).str.startswith("FAIL").sum())
    dil = rd("validation", "optical_validation.csv")
    comp = rd("validation", "optical_composition.csv")
    sweep = rd("validation", "decoder_validation.csv")
    keys = rd("validation", "key_sensitivity.csv")
    bench = rd("validation", "benchmark_comparison.csv")
    dil_abs = float(dil["choi_trace_norm_error"].max())
    dil_cond = float(dil["choi_conditional_error"].max())
    spread = float(dil["acceptance_spread"].max())
    comp_err = float(comp["choi_trace_norm_error"].max()) if comp is not None else np.nan
    pvd = float(sweep.loc[sweep["method"] == "pauli_vs_dm_residual",
                          "Fe_exact"].max())
    dmg = sweep[sweep["method"] == "density_matrix"]
    gap_max = float(dmg["gap_Fe_minus_C"].max())
    gap_min = float(dmg["gap_Fe_minus_C"].min())
    bench_max = float(bench["relative_error"].max()) if bench is not None else np.nan

    def key_spread(path):
        kr = keys[(keys["key_set_id"] == "random_range") &
                  (keys["noise_path"] == path)].dropna(subset=["Fe_min"])
        return float((kr["Fe_max"] - kr["Fe_min"]).max())
    ks_base, ks_xpm = key_spread("baseline"), key_spread("xpm_dominant")
    gx = sweep[(sweep["noise_path"] == "xpm_dominant") &
               (sweep["method"] == "density_matrix")]
    gi = gx[gx["key_case"] == "identity"].sort_values("q")["Fe_exact"].to_numpy()
    gn = gx[gx["key_case"] == "fixed_nonclifford_ry"].sort_values("q")["Fe_exact"].to_numpy()
    nc_vs_id = float(np.abs(gn - gi).max())
    fwm_bench = s1[s1["validation_case"].astype(str).str.contains("FWM.*benchmark",
                                                                  regex=True)]
    fault_rows = s1[s1["validation_case"].astype(str).str.contains("fault injection")]

    # -------------------------------------------------------- physical scans
    pts = point_summary()
    p_main = pts[pts["run_id"] == MAIN].iloc[0]
    mp = rd("illustrative_baseline", "operating_map.csv")
    n_ok = int(mp["quality_domain_nonempty"].sum()); n_tot = len(mp)
    n_cert = int(mp["W_q_nonempty"].sum())
    y_cert_min = float(mp.loc[mp["quality_domain_nonempty"].astype(bool),
                              "Y_lb_at_star"].min())
    n_unc_ub = int(((~mp["quality_domain_nonempty"].astype(bool)) &
                    (mp["max_Y_ub_over_Omega_q"] > Y_min)).sum())
    bnd = rd("illustrative_baseline", "certification_boundary.csv")
    bnd_ok = (bnd[bnd["status"].astype(str).str.startswith("converged")]
              if bnd is not None else None)
    bnd_res = (float(bnd_ok["residual_in_C"].max())
               if bnd_ok is not None and len(bnd_ok) else np.nan)
    bnd_unres = (int(bnd["status"].astype(str).str.startswith("UNRESOLVED").sum())
                 if bnd is not None else 0)
    bnd_multi = (int((bnd["n_sign_changes_on_the_sampled_grid"] > 1).sum())
                 if bnd is not None else 0)
    pw = rd("illustrative_baseline", "fwm_profile_width.csv")
    scn = rd("illustrative_baseline", "scenario_sensitivity.csv")
    rob = rd("illustrative_baseline", "robustness_summary.csv")
    mr = pd.concat([x for x in (rd(r, "mode_reference.csv") for r, *_ in PTS)
                    if x is not None], ignore_index=True)
    conv = pd.concat([x for x in (rd(r, "convergence.csv") for r, *_ in PTS)
                      if x is not None], ignore_index=True)
    conv_y = float(conv["relative_change_Y_lb"].max())
    conv_c = float(conv["absolute_change_C"].replace(0.0, np.nan).max())
    fwmsplit = rd(MAIN, "fwm_slot_split.csv")
    occ_pct = 100 * float(fwmsplit["power_fraction_occupied"].iloc[0])
    vac_pct = 100 * float(fwmsplit["power_fraction_vacant_in_band"].iloc[0])
    n_vac = int(fwmsplit["n_vacant_in_band_slots"].iloc[0])

    sc_c = scn[scn["anchor"] == SCN_ANCHOR]
    failing = sc_c.loc[~sc_c["Omega_F_nonempty"].astype(bool), "label"].tolist()
    M_fail = ", ".join(f"`{r.label}` (M = {r.max_C_over_Omega_q:.4f})"
                       for r in sc_c.itertuples() if r.label in failing) or "none"
    fxr = rob[(rob["anchor"] == SCN_ANCHOR) &
              (rob["mode"] == "fixed nominal wavelength")].set_index("scenario")
    n_fixed_cert = int(fxr["certified"].astype(bool).sum())
    fixed_fail = sorted(x for x in fxr.index if not bool(fxr.loc[x, "certified"]))
    extra_fail = [x for x in fixed_fail if x not in failing]
    fixed_vs_empty = (
        "its failures are exactly the scenarios whose certified domain is empty, "
        "so at this operating point holding the wavelength fixed loses nothing "
        "relative to retuning" if not extra_fail and set(fixed_fail) == set(failing) else
        "it also fails under " + ", ".join(f"`{x}`" for x in extra_fail) +
        ", where retuning still finds a certified candidate")
    robrow = rob[(rob["anchor"] == SCN_ANCHOR) &
                 (rob["mode"] == "robust selection over the enumerated scenario set")]
    rob_empty = bool(pd.isna(robrow["lambda_nm"].iloc[0])) if len(robrow) else True

    pw_m = (pw[(pw["length_km"] == 5.0) & (pw["launch_dbm"] == -10.0)]
            .sort_values("fwm_profile_rms_ghz") if pw is not None else None)
    pk_ratio = float(pw_m["peak_nu_FWM_ratio_vs_delta_line"].iloc[-1])
    nu_star_delta = float(pw_m["nu_FWM_at_star"].iloc[0])
    nu_star_wide = float(pw_m["nu_FWM_at_star"].iloc[-1])
    share_wide = float(pw_m["fwm_share_of_nu_at_star"].iloc[-1])
    dY_wide = float(pw_m["relative_Y_lb_change_vs_delta_line"].iloc[-1])
    dOmF_wide = int(pw_m["n_Omega_F_change_vs_delta_line"].iloc[-1])

    nL = int(mp["length_km"].nunique()); nP = int(mp["launch_dbm"].nunique())
    lbp = rd("illustrative_baseline", "link_budget_points.csv")
    lbc = rd("illustrative_baseline", "link_budget_curve.csv")
    bes = rd("illustrative_baseline", "base_error_sweep.csv")
    rx_cfg = cfg.get("classical_link", {}).get("receivers", [])
    anchors = cfg.get("base_error_sweep", {}).get("anchors", [])
    lb_txt, e0_txt = "", ""
    if lbc is not None and len(rx_cfg):
        parts = []
        for rx in rx_cfg:
            g = lbc[lbc["receiver"] == rx["id"]].sort_values("length_km")
            ok_y = g[g["Y_lb"] >= Y_min]
            ok_f = g[g["n_Omega_F"] > 0]
            parts.append(
                f"{rx['data_rate_gbps']:g} Gb/s receiver ({rx['rx_sensitivity_dbm']:g} dBm): "
                f"fidelity certified up to {fmt(ok_f['length_km'].max(), 3)} km and "
                f"Y_min met up to {fmt(ok_y['length_km'].max(), 3)} km")
        lb_txt = "; ".join(parts)
    if lbp is not None:
        pm = lbp[lbp["point_kind"] == "predeclared"]
        e0_main = float(pm.loc[(pm["length_km"] == 5.0) & (pm["launch_dbm"] == -10.0),
                               "e0_max_for_certification"].iloc[0])
        e0_txt = f"{100*e0_main:.3f}%"
    else:
        e0_main = np.nan
    reg_main = rd(MAIN, "certified_regions.csv")
    wq_n_band = wq_n_iso = 0
    if reg_main is not None:
        wq = reg_main[(reg_main["region"] == "W_q") & (reg_main["n_intervals"] > 0)]
        wq_n_band = int((wq["width_nm"] > 1e-9).sum())
        wq_n_iso = int((wq["width_nm"] <= 1e-9).sum())
    dT = float(mr["T_relative_difference"].abs().max())
    dnu = float(mr["nu_relative_difference"].abs().max())
    dC = float(mr["C_absolute_difference"].abs().max())
    dY = float(mr["Y_lb_relative_difference"].abs().max())

    # margin comparison at the four points
    mtxt = []
    for r in pts.itertuples():
        if r.run_id not in ZERO:
            continue
        if np.isfinite(r.lam) and np.isfinite(r.lam0):
            dlam = r.lam - r.lam0
            dyl = r.Ylb / r.Ylb0 - 1.0 if r.Ylb0 > 0 else np.nan
            mtxt.append(f"| {r.length_km:g} km, {r.launch_dbm:+.0f} dBm/ch | "
                        f"{r.lam0:.4f} | {r.lam:.4f} | {dlam:+.4f} | "
                        f"{100*dyl:+.3f}% |")
        else:
            mtxt.append(f"| {r.length_km:g} km, {r.launch_dbm:+.0f} dBm/ch | "
                        f"{fmt(r.lam0, 8)} | {fmt(r.lam, 8)} | undefined | "
                        f"undefined |")
    margin_table = "\n".join(mtxt)

    # ---------------------------------------------------------- changes.md ---
    ch = f"""# Changes

Corrections applied to the earlier version of this code before any physical
scan was rerun, with the mathematical basis of each and the results that
changed.  Every number below is read from the regenerated outputs.

## 1. Four-wave-mixing normalization

**What was wrong.**  Pump pairs were enumerated with `i <= j` and given the
amplitude degeneracy `d = 3` (i == j) and `d = 6` (i != j) while the
*conventional* fibre coefficient `gamma = 2 pi n2 / (lambda A_eff)` was used
unchanged.

**Basis.**  Expanding `dA/dz = i gamma |A|^2 A` for a multi-frequency field and
counting each unordered pump pair once gives amplitude coefficients 1 and 2 with
the conventional gamma; the 3 / 6 bookkeeping belongs with `gamma/3`.  Using
both overcounts the generated amplitude by 3 and the power by 9.

**Fix.**  `src/fwm.py` takes an explicit `convention`: `conventional_gamma`
(`d = 1, 2`, the production setting) or `gamma_over_three` (`d = 3, 6` with
`gamma/3`).  The choice is recorded in the audit block and in Main Table 1.

**Independent tests.**  Lossless, phase-matched seeded benchmarks that do not
call the production prefactor: degenerate `P = gamma^2 P_p^2 P_s L^2` and
nondegenerate `P = 4 gamma^2 P_i P_j P_k L^2`, plus `L^2`, `gamma^2` and cubic
launch-power scalings, and a fault injection in which the earlier convention
gives exactly nine times the benchmark power and is rejected ({len(fwm_bench)} benchmark
rows and {len(fault_rows)} fault-injection row in Table S1; tests
`test_fwm_degenerate_benchmark_and_factor_of_nine_rejection`,
`test_fwm_nondegenerate_benchmark`).

**What changed.**  Every four-wave-mixing occupancy falls by a factor of 9 at
fixed launch power, which enlarges the certified sets and moves the
certification boundary.  The effect cannot be isolated from the present
results, because the receiver convention, the tuning margin and the candidate
grid were corrected in the same revision.

## 2. Parameter tables generated from the executed configuration

`src/runio.py:provenance_rows` no longer holds hard-coded physical values: it
reads the effective configuration of the run, so the exported table cannot
drift away from the numbers the simulation used.  The mismatches found in the
earlier table are gone by construction:

| Input | Earlier table | Executed configuration (now exported) |
|---|---|---|
| Attenuation | analytic, 0.190 dB/km anchor | tabulated SMF-28 table, 0.180 dB/km at 1550 nm |
| Filter order | 3 | {int(cfg['link']['filter_order'])} |
| XPM differential coupling | 0.05 | {cfg['optics']['xpm_kappa']:g} |
| Relative power fluctuation | 10% rms | {100*cfg['optics']['xpm_rel_fluct']:.0f}% rms |
| XPM fluctuation bandwidth | 10 GHz | {cfg['optics']['xpm_band_ghz']:g} GHz |

Value, provenance category, reference plane and sensitivity range are separate
columns.  Interpolated datasheet rows are labelled as interpolations, not as
measurements and not as bounds.

## 3. Tolerances and validation counts

The validation helper accepted a generic dimensionful tolerance, so a
coefficient of order 1e-23 passed on the absolute criterion whatever its
relative error.  Tolerances are now relative wherever the reference is nonzero
and absolute only for zero limits and dimensionless residuals; a check with no
stated tolerance raises.  `test_raman_calibration_check_rejects_a_wrong_
normalization` shows a coefficient inflated by 1e6 failing the corrected
comparison and passing the earlier one.

Counts are reported separately: **{n_assert} asserted checks ({n_fail} failures)
and {n_diag} report-only diagnostics**, with sweep sizes quoted per stage.  The
earlier count mixed these two categories.

## 4. Receiver model

The signal used a power average `int |psi|^2 T_q df` while the noise was
projected into a fixed Gaussian mode.  Both now use the same fixed-mode
projection `eta_mode = |int |psi|^2 t(f) df|^2` with `t = sqrt(T_q)`.  For the
10 GHz mode and the 25 GHz order-4 filter the two differ by 0.991466 versus
0.990210.  The power average is retained only as a declared scenario
(`receiver_power_average`) and is never mixed into a main result.

`src/mode_reference.py` adds an independent full-chain spectral reference
(`T_ref`, `nu_ref`) that keeps the frequency dependence through all three
traversals and projects once at the end.  Against the scalar recurrence it
differs by up to {100*dT:.2f}% in `T`, {100*dnu:.2f}% in `nu`, {dC:.1e} in `C`
and {100*dY:.1f}% in `Y_lb`.  The main results keep the scalar recurrence, whose
physical content is that each terminal re-launches into the specified mode: the
per-traversal modal overlap is the declared cost of that restoration.  No free
mode restoration is introduced, and the size of the difference is reported.

## 5. FWM spectral profile

Generated components were delta lines.  `modes.occupancy_from_profiled_lines`
integrates a normalized profile of prescribed rms width against the receiver
mode, with the delta line as the exact zero-width limit (verified against
`occupancy_from_line` and against the analytic Gaussian convolution).  Widths of
0, 1, 5 and 25 GHz are compared at all four predeclared operating points.

## 6. Candidate domain, margin and selectors

* The effective guard is `g + m`, and candidate boundary points are generated at
  `g + m`: a zero-width component at exactly `g` does not survive a nonzero
  margin.  The main configuration uses `g = {guard:g} GHz`, `m = {margin:g} GHz`.
* The predeclared fixed reference is an EXACT rule: if that wavelength is not an
  allowed candidate the rule is reported invalid, and the nearest allowed
  candidate is a separate, separately labelled comparator.
* Exact identity-key fidelity is computed for every allowed candidate by Pauli
  enumeration, not at a handful of points.
* Grid refinement and quadrature refinement are varied separately.

## 7. Boundary solve

The certification boundary is a root of `max_lambda C - F_min`, found by
bracketed bisection on directly re-evaluated maxima.  Every sign change on the
sampled grid is solved separately (monotonicity is not assumed) and the subset
used for the inner wavelength maximization is verified against the full allowed
set at the root.  Worst residual over the converged powers {fmt(bnd_res, 2)} in
`C` against a 1e-6 target; {bnd_unres} unresolved; {bnd_multi} powers with more
than one sign change.

## 8. What the corrections changed in the reported results

| Quantity | Earlier version | This version |
|---|---|---|
| FWM power normalization | nine times too large | benchmarked and corrected |
| Receiver signal transmission | power average 0.991466 | modal overlap 0.990210 |
| Candidate domain | guard 50 GHz, no margin | guard {guard:g} GHz + {margin:g} GHz margin |
| Exact fidelity curve | <= 80 sampled candidates | every allowed candidate |
| Fixed reference 1550 nm | snapped to the nearest allowed candidate | reported invalid, nearest allowed listed separately |
| Certification boundary | interpolated, verified | bracketed bisection, residual {fmt(bnd_res, 2)} in C |
| Validation counting | mixed single count | {n_assert} asserted + {n_diag} diagnostics |

## 9. Submission reproducibility update

Finite-width FWM integration now uses the analytic product of the normalized
line profile and accepted Gaussian mode before quadrature. The physical
integrand, profile powers and delta-line branch are unchanged. Direct adaptive
quadrature regression checks include the filter and photon-energy factor.

Interval summaries on compressed allowed grids retain the original full-grid
indices. Excluded carrier guards therefore separate intervals even when every
allowed candidate is certified; this changes counts, not physical predictions.

The full workflow now exports the 6 km, -10 dBm/ch example in Main Table 2.
The original four diagnostic anchors and their margin comparisons are retained.
The -24 dBm receiver benchmark is labelled as an illustrative assumption,
matching the manuscript; its numerical value is unchanged.
"""
    open(os.path.join(RES, "changes.md"), "w", encoding="utf-8").write(ch)
    print("  wrote changes.md")

    # -------------------------------------------------------- validation ----
    ptxt = []
    for r in pts.itertuples():
        if np.isfinite(r.lam):
            ptxt.append(
                f"| {r.length_km:g} km, {r.launch_dbm:+.0f} dBm/ch | {r.role} | "
                f"{r.n_allowed} | {r.n_Omega_F} | {r.lam:.4f} | {r.C:.6f} | "
                f"{r.Fe:.6f} | {r.Ylb:.4e} | {r.status} |")
        else:
            ptxt.append(
                f"| {r.length_km:g} km, {r.launch_dbm:+.0f} dBm/ch | {r.role} | "
                f"{r.n_allowed} | 0 | undefined | max C = {r.max_C:.6f} | "
                f"undefined | undefined | NOT certified: Omega_F empty |")
    ptable = "\n".join(ptxt)

    vr = f"""# Validation report

What the numerics verify, and what the physical model merely assumes.  The
optical scenario is **illustrative and uncalibrated**: no measured fibre
spectrum, receiver characterization or deployment record was supplied.

## 1. Numerical verification (independent of any fibre measurement)

{n_assert} asserted checks, {n_fail} failures, {n_diag} report-only diagnostics.
Sweep sizes are separate: {len(dil)} optical dilation points, {len(sweep)}
decoder sweep rows, {len(keys)} key-sensitivity rows.

* **Accepted optical channel.**  An independent thermal-environment
  beam-splitter dilation, exact within each total-photon-number sector,
  reproduces `c0 = T/(1+nu)^4` and `c1 = 2 nu (1+nu-T)/(1+nu)^4` over
  {len(dil)} (T, nu) points: maximum accepted-Choi trace-norm error
  {dil_abs:.1e} (target 1e-9) and maximum normalized-channel error
  `|| J_num/s_num - J_ref/s_ref ||_1` = {dil_cond:.1e} (target 1e-7).
  Acceptance is input independent to {spread:.0e} over six polarization probe
  states; positivity, trace nonincrease and `Tr_out J = s I/2` hold.
* **Composition.**  Three explicit Fock-space traversals with final-only
  acceptance reproduce the single effective channel to {comp_err:.0e},
  confirming `nu = eta_3 eta_2 nu_1 + eta_3 nu_2 + nu_3`.
* **Logical recovery.**  Exact enumeration of all 4^7 net Pauli strings and an
  independent 8-qubit density-matrix recovery agree to {pvd:.0e}.  Across the
  sweep `F_e >= C_7,1(q)` holds, maximum gap {gap_max:.3f}, most negative
  residual {gap_min:.0e}.  The synthetic benchmark is reproduced to
  {bench_max:.0e} relative.
* **Four-wave mixing.**  Independent low-conversion benchmarks that do not call
  the production prefactor confirm the corrected normalization; the earlier
  convention is rejected at nine times the benchmark power.

This verifies the mathematics of the reduction and the recovery.  It does not
show that a physical fibre background is thermal, independent, or of the assumed
spectral shape.

## 2. Physical assumptions (conditional; not verified here)

1. The Raman spectrum is the Hollenbeck-Cantrell (2002) intermediate-broadening
   fit to the measured gain of fused silica, with the Bose factor applied on each
   side and the level fixed at the measured coefficient of Qi et al. (2010,
   -0.5995 THz anti-Stokes).  The polarization convention is resolved from that
   source (Eqs. 5 -> 6 -> 15): the coefficient is polarization summed and the
   factor 1/2 is applied once, in the occupancy integral.  An independent
   near-carrier measurement (da Silva et al. 2014) lies a factor of about 4
   below this level at 0.6 THz and grows linearly with offset where the model is
   nearly flat; it is carried as the scenario `raman_level_dasilva2014`.
2. The aggregate background is treated as an independent isotropic thermal
   effective state.  Adding Raman, FWM and leakage powers does not establish
   thermal photon statistics, nor independence across qubits and traversals.
3. The receiver is a fixed-mode projection with the declared mode restoration at
   every traversal output.  The full-chain spectral reference differs by up to
   {100*dT:.2f}% in `T` and {100*dY:.1f}% in `Y_lb`: a model choice, not a
   numerical error.
4. FWM components are delta lines in the main results; prescribed profiles of
   1, 5 and 25 GHz rms are phenomenological, not a modulated-FWM simulation.
5. The XPM model is a flat two-sided power-fluctuation PSD of
   {100*cfg['optics']['xpm_rel_fluct']:.0f}% rms depth over
   {cfg['optics']['xpm_band_ghz']:g} GHz with differential coupling
   kappa = {cfg['optics']['xpm_kappa']:g}, converted to a Gaussian phase.
   Gaussian phase statistics are assumed, not derived from an on-off keyed
   intensity, and phase uniformity over the accepted pulse is assumed.
6. Backward FWM generation, reflection and Rayleigh backscatter are not modeled,
   so the return traversal receives no modeled FWM.
7. The {margin:g} GHz tuning margin is an illustrative scenario choice, not a
   measured hardware tolerance; 2.5 GHz is included as a sensitivity case,
   motivated by the EXFO FTBx-2850 frequency uncertainty specification.
8. Only one datasheet anchor (1550 nm) lies inside the quantum band; the rest of
   the attenuation curve there is interpolated, and a manufacturer maximum is
   neither a measurement nor a bound at interpolated wavelengths.
9. The classical launch power at the link-budget minimum uses the literature
   sensitivity at 1.25 Gb/s and the assumed -24 dBm benchmark at 10.7 Gb/s,
   the worst classical attenuation, and assumed 3 dB output demultiplexer
   and 3 dB system margin terms. Powers are at the fiber input; the 3 dB
   input multiplexer loss applies only to the upstream transmitter power.

## 3. Sensitivity

Of the enumerated scenarios at 5 km / -10 dBm/ch, {len(failing)} empty the
quality-certified domain: {M_fail}.  Classification uses the full-domain maximum
`M_theta = max_(s>0) C_7,1(q)`; `M_theta < F_min` means no candidate is certified
by this bound, not that exact fidelity fails.  The nominal wavelength held
fixed is certified under {n_fixed_cert} of {len(fxr)} scenarios; {fixed_vs_empty}.
The intersection of the certified domains over all scenarios is
{"empty" if rob_empty else "nonempty"}.

Prescribed FWM broadening lowers the peak occupancy (x{fmt(pk_ratio, 3)} at
25 GHz rms) but raises the occupancy at the selected wavelength from
{nu_star_delta:.1e} to {nu_star_wide:.1e} ({100*share_wide:.2f}% of nu there),
because that candidate is fed only by profile tails.  The certified set changes
by {dOmF_wide} candidates and `Y_lb` by {100*dY_wide:+.4f}%, so the narrow-line
approximation does not drive the conclusions at these parameters; the tails are
qualitatively important and quantitatively small here.

Tuning margin, at the four predeclared points:

| Operating point | lambda* at 0 GHz margin | lambda* at {margin:g} GHz margin | displacement (nm) | relative change in Y_lb |
|---|---|---|---|---|
{margin_table}

## 4. Convergence

Grid refinement (2 -> 1 -> 0.5 GHz at fixed quadrature) and quadrature
refinement (Gauss-Hermite 48 -> 96 at fixed grid) were varied separately at all
four predeclared points.  Worst relative change in the optimized `Y_lb`
{conv_y:.1e} (target 1e-3); worst absolute change in `C` at the reported
selections {fmt(conv_c, 2)} (target 1e-6).  The certification boundary is
resolved to {fmt(bnd_res, 2)} in `C`.

## 5. Unresolved

* No Raman spectrum, receiver characterization or attenuation measurement was
  taken for this fibre; the scans use published models and datasheet values, so
  every physical scan is conditional on them.
* A physically derived FWM profile needs the modulation and coherence model of
  the classical carriers; prescribed widths only bound the direction of the
  effect.
* Independence of noise realizations across qubits and traversals is assumed;
  the correlated-block control is a diagnostic, not a proof.
* The sampled key range is not a worst-case guarantee over private keys and does
  not measure Pauli-twirl error.

## 6. Manuscript operating-point comparisons

| Operating point | Role | allowed | Omega_F | lambda* (nm) | C | F_e (identity) | Y_lb | status |
|---|---|---|---|---|---|---|---|---|
{ptable}
"""
    open(os.path.join(RES, "validation_report.md"), "w", encoding="utf-8").write(vr)
    print("  wrote validation_report.md")

    # ------------------------------------------------------------- captions --
    agree = []
    for rule, nice in (("max_T_over_Omega_q", "maximum transmission"),
                       ("min_q_over_Omega_F", "minimum q")):
        lam_r = p_main[f"lam_{rule}"]
        if np.isfinite(lam_r):
            same = abs(lam_r - p_main.lam) < 1e-9
            agree.append(f"{nice.capitalize()} selects {lam_r:.4f} nm "
                         f"(C = {p_main[f'C_{rule}']:.6f}, "
                         f"{'the same wavelength' if same else 'a different wavelength'};"
                         f" {p_main[f'status_{rule}']})")
    agree_txt = "; ".join(agree) if agree else "no comparator is defined"

    cap = f"""# Figure captions

Standalone captions.  Every setting named here is the one the run actually used;
every number is read from the stored outputs.

**Figure 1 | Optical-channel verification.**  Ideal final one-photon acceptance
on two polarization modes, each carrying an independent isotropic thermal
background, on synthetic (T, nu) pairs.  (a) Acceptance `s` and (b) conditional
mixing `r` against the per-polarization final occupancy nu at T = 0.05, 0.2 and
0.9.  Curves are the closed-form weights `c0 = T/(1+nu)^4` and
`c1 = 2 nu (1+nu-T)/(1+nu)^4`; markers are an independent thermal-environment
beam-splitter dilation, built exactly inside each total-photon-number sector with
both truncated environments traced out before projection.  The pure-loss limit
nu -> 0 gives `s = T` and `r = 0` exactly and cannot appear on a logarithmic
horizontal axis.  (c) Accepted-Choi trace-norm discrepancy
`||J_num - J_formula||_1` against the discarded thermal probability for four
(T, nu) pairs, each with its own colour and marker shape.  The **red star on
each series marks the production setting** (environment tail tolerance 1e-16),
which every reported run uses; the remaining points are deliberately loosened
cutoffs, shown only to exhibit the error tracking the discarded tail.  The
**black dashed line is the 1e-9 target tolerance**.  Over all
{len(dil)} sweep points the achieved discrepancy is {dil_abs:.1e} and the
normalized-channel error is {dil_cond:.1e}.  The three-traversal composition
check and the receiver-model comparison are in Supplementary Figure S1.  An
independent numerical construction reproduces the adopted accepted optical
channel over the tested domain; no claim about measured fibre noise follows.

**Figure 2 | Logical fidelity and the correction bound.**  [[7,1,3]] Steane code
with the fixed CSS Hamming recovery.  Both noise paths are one-parameter
families indexed by u (0 plus 33 logarithmic values from 1e-3 to 10) at fixed
transmission eta = 0.55 per pass; variances are scaled first and converted to z
afterwards.  *Baseline* at u = 1: nu = 3e-4 per polarization, e_0 = 0.002,
sigma_phi^2 = 4e-5 on the two co-propagating passes and 0 on the
counter-propagating pass.  *Strong XPM* at u = 1: nu = 3e-5, e_0 = 2e-4,
sigma_phi^2 = 0.10 on all three passes.  (a) Exact logical infidelity `1 - F_e`
against the affected-location probability q on the baseline path, with
`1 - C_7,1(q)` as the upper bound on the infidelity; identity and fixed
non-Clifford keys `A_j = R_y(0.628)`, `B_j = R_y(1.047)` are density-matrix
results and the circles are an independent enumeration of all 4^7 net Pauli
strings for identity keys.  (b) Bound gap `F_e - C_7,1(q)` on both paths (inset:
baseline only, logarithmic).  The gap is nonnegative everywhere, maximum
{gap_max:.3f}, most negative value {gap_min:.0e}; it reflects support counting,
cancellation inside affected histories, and errors outside the guaranteed set
that the recovery also corrects.  (c) Sampled key dependence: the range
max-min over 16 reproducible commuting random key sets (markers) and the
absolute difference between the fixed non-Clifford and identity cases (dashed);
the q = 0 point carries no key dependence and is omitted.  The shaded band is the
measured enumeration versus density-matrix residual ({pvd:.0e}), a numerical
reference, not an error bar for every non-Clifford computation.  The baseline
range reaches {ks_base:.0e}; the strong-XPM range reaches {ks_xpm:.3f} and the
fixed non-Clifford case differs from identity keys by up to {nc_vs_id:.3f}.  The
random key rule draws a common axis per location from x, y, z and both angles
from [0, 2 pi): not Haar sampling, so the sampled range is not a worst-case
guarantee over private keys, and key dependence alone does not measure
Pauli-twirl error.

**Figure 3 | Wavelength selection.**  Predeclared operating point: 5 km per
traversal, -10 dBm per classical channel; 35 fixed carriers on a 100 GHz grid
from 191 600 to 195 900 GHz launched A->B, so the three quantum traversals are
co-, counter- and co-propagating.  Guard {guard:g} GHz with an added tuning
margin of {margin:g} GHz (a declared scenario choice, not a hardware
measurement); candidates on a {cfg['search']['grid_step_ghz']:g} GHz grid with
the exact hardware, guard and search boundary points inserted; accepted mode
Gaussian of {cfg['link']['mode_fwhm_ghz']:g} GHz FWHM; receiver filter
super-Gaussian of order {int(cfg['link']['filter_order'])},
{cfg['link']['filter_fwhm_ghz']:g} GHz FWHM and
{cfg['link']['filter_isolation_db']:g} dB floor; insertion transmission
{cfg['link']['eta_insertion_per_pass']:g} per pass split symmetrically; base
error {cfg['link']['e0_per_pass']:g} per pass; fixed-mode receiver convention.
(a) Final propagated Raman, FWM and leakage occupancies PER POLARIZATION, each
attenuated by the later traversals; FWM components are delta lines (narrow-line
limit).  (b) Single-mode
acceptance `s`: its peaks are background replacement photons, not improved
quantum transmission.  (c) The lower-bound certificate `C_7,1(q)` and the exact
logical fidelity `F_e` for identity keys, computed by Pauli enumeration at every
allowed candidate, against `F_min = {F_min}`.  Certification under the lower
bound is a different statement from the exact fidelity of one key set.
(d) `Y_lb`, `Y_ub = s^7` and the exact `Y = s^7 F_e` for identity keys, with
`Y_min = {Y_min:g}`; the colours pair `Y_lb` with `C` and `Y` with `F_e` in (c).
The shaded bands are the certified region `W_q` ({wq_n_band} finite-width
component{"s" if wq_n_band != 1 else ""}, {wq_n_iso} zero-width isolated
point{"s" if wq_n_iso != 1 else ""}; isolated points would be drawn as thin
stems).  Vertical guide lines through all panels mark the comparator
selections, and coincident selections share one line.  Here the
constrained maximum-`Y_lb` rule selects {p_main.lam:.4f} nm with
`s = {p_main.s:.4f}`, `C = {p_main.C:.6f}`, `Y_lb = {p_main.Ylb:.4e}` and exact
`F_e = {p_main.Fe:.6f}`.  The selected wavelength maximizes the LOWER BOUND over
the certified set; it need not maximize the exact yield.  Selector agreement is
reported as found: {agree_txt}.  The predeclared fixed reference
{cfg['targets']['fixed_reference_nm']:g} nm is {p_main.fixed_status}; the
nearest allowed candidate ({fmt(p_main.near_lam, 8)} nm) is a distinct rule.
The Raman spectrum is the published Hollenbeck-Cantrell fit to measured silica
gain, normalized at the measured coefficient of Qi et al.; the aggregate
background is assumed isotropic thermal.

**Figure 4 | Operating regions and sensitivity.**  Same fixed classical
allocation, receiver assumptions and targets; the quantum wavelength is
re-optimized at every point, with guard {guard:g} GHz and {margin:g} GHz added
margin in panels (a), (b) and (d).  (a) `log10 max_Omega_F Y_lb` over {nL} lengths
by {nP} launch powers on a 1 km by 1 dB sampled grid; grey marks the
{n_tot - n_ok} of {n_tot} sampled points where the quality-certified domain is
empty.  Undefined optima are masked, never zero-filled.
{"A Y_min contour is drawn where optimized yields cross it." if y_cert_min < Y_min else f"No Y_min contour is drawn: every certified sampled optimum exceeds Y_min, the smallest being {y_cert_min:.2e}. That statement concerns the optimized wavelength at the sampled operating points, not every wavelength at those points."}
The black curve is the fidelity-feasibility boundary `max_lambda C = F_min`,
located by bracketed bisection on directly re-evaluated maxima (worst residual
{fmt(bnd_res, 2)} in `C`; monotonicity in length is not assumed).  The dashed
and dotted black lines are the minimum launch power that closes the classical
link for the {" and ".join(f"{r['data_rate_gbps']:g} Gb/s" for r in rx_cfg)}
receivers (sensitivity, worst classical attenuation, multiplexer, demultiplexer
and margin); points above a line close that link.  (b) Selected wavelength on
the same plane, masked where the optimum is undefined, with the same link-budget
lines.
(c) Tuning-margin sensitivity at the predeclared interior point (3 km,
-13 dBm/ch) and the original comparison point (5 km, -10 dBm/ch): wavelength
displacement (upper) and relative change in the optimized `Y_lb` (lower) for
margins 0, 1, 2, 2.5 and 5 GHz at the 50 GHz guard.  Ratios
are computed only where both compared optima exist; a red cross marks a margin
at which the certified set is empty.  (d) Scenario sensitivity at 5 km,
-10 dBm/ch: bars are `M_theta = max_(s>0) C_7,1(q)` with the wavelength free,
red exactly when `M_theta < F_min`, which means no candidate is certified by this
bound over the evaluated domain, NOT that exact fidelity fails; open diamonds are
the certificate at the nominal wavelength held fixed.  Scenario ranges are
declared sensitivity choices, not confidence intervals.

**Supplementary Figure S1 | Composition check, receiver-model comparison and FWM
profile widths.**  (a) Trace-norm discrepancy between three explicit Fock-space
traversals with final-only acceptance and the single effective (T, nu) channel.
(b) Relative difference of the full-chain spectral reference from the scalar
recurrence, (full chain / scalar) - 1 in percent, for `T`, `nu` and `Y_lb` at
the evaluated candidates of the four predeclared points;
the main results use the scalar recurrence with the declared mode restoration at
each traversal output, and this panel is the size of that modelling choice.
(c) Peak summed per-traversal FWM occupancy over the allowed domain against the prescribed
profile rms width; the widths are prescribed, not derived from a modulation
model.

**Supplementary Figure S2 | Base-error requirement and the classical link
budget.**  (a) Largest per-traversal base error `e_0` for which some allowed
candidate still meets `C_7,1(q) >= F_min`, over the operating map of Fig. 4;
grey marks points where no candidate is certified even at `e_0 = 0`.  White
contours are the published `e_0` anchors that fall inside the plotted range
(marked on the colour bar), with the same line styles as in (b).  Black lines
are the link-budget minimum launch power of each receiver, as in Fig. 4.
(b) The same quantity along each link-budget line (orange circles, solid line:
1.25 Gb/s; green squares, dashed line: 10.7 Gb/s; markers are staggered so that
coincident curves stay visible).  Grey horizontal lines are the anchors, named on the
right axis: {"; ".join(f"{100*float(a['e0']):g}%, {a['source']}" for a in anchors)}.
A receiver is certified at the nominal `e_0` wherever its curve lies above the
`nominal` line.  (c) The optimized lower yield bound `Y_lb` along each
link-budget line, with the same markers at the nominal `e_0 = {100*cfg['link']['e0_per_pass']:g}%`,
with `Y_min` dashed.  {lb_txt + "." if lb_txt else ""}  At the 5 km,
-10 dBm/ch point the largest admissible `e_0` is {e0_txt or "undefined"}.
"""
    open(os.path.join(RES, "captions.md"), "w", encoding="utf-8").write(cap)
    print("  wrote captions.md")

    # ------------------------------------------------- results evidence -----
    ev = f"""# Results evidence outline

One short paragraph per main figure, using only quantities from the regenerated
outputs.  Numerical verification and model-conditional prediction are kept
apart.

**Figure 1 (verification).**  An independent thermal-environment dilation
reproduces the derived accepted optical operation over {len(dil)} synthetic
(T, nu) points: maximum accepted-Choi trace-norm error {dil_abs:.1e} against a
1e-9 target, maximum normalized-channel error {dil_cond:.1e} against 1e-7, and
acceptance independent of the input state to {spread:.0e} over six polarization
probe states.  Three explicit Fock-space traversals with final-only acceptance
reproduce the single effective channel to {comp_err:.0e}.  These are statements
about the mathematics of the reduction, not about a physical fibre.

**Figure 2 (verification).**  With an explicit Steane encoder and fixed CSS
Hamming recovery, the exact conditional logical entanglement fidelity satisfies
`F_e >= C_7,1(q)` at every tested point on both noise paths; the largest gap is
{gap_max:.3f} and the most negative residual {gap_min:.0e}.  Two independent
exact methods agree to {pvd:.0e}.  Key dependence is small on the baseline path
({ks_base:.0e} over 16 random commuting key sets) and significant on the
strong-XPM path ({ks_xpm:.3f}; the fixed non-Clifford set differs from identity
keys by {nc_vs_id:.3f}), which is why the general density-matrix calculation is
used.  This demonstrates key dependence; it does not quantify Pauli-twirl error.

**Figure 3 (model-conditional).**  At 5 km per traversal and -10 dBm per
classical channel, with a {guard:g} GHz guard and {margin:g} GHz added margin,
{p_main.n_Omega_F} of {p_main.n_allowed} allowed candidates are certified by the
lower bound.  The constrained maximum-`Y_lb` rule selects {p_main.lam:.4f} nm
with `s = {p_main.s:.4f}`, `q = {p_main.q:.4g}`, `C = {p_main.C:.6f}`,
`Y_lb = {p_main.Ylb:.4e}` and exact identity-key `F_e = {p_main.Fe:.6f}`
(`Y = {p_main.Yex:.4e}`).  {agree_txt}.  The predeclared fixed reference
{cfg['targets']['fixed_reference_nm']:g} nm is {p_main.fixed_status}.  FWM
products lie on the classical grid, with {occ_pct:.0f}% of the modeled product
power on occupied carriers and {vac_pct:.1f}% on the {n_vac} vacant in-band
slots; only vacant positions can coincide with an allowed candidate.  All of
this is conditional on the declared Raman shape, receiver model and background
assumptions.

**Figure 4 (model-conditional).**  Over a sampled map of {n_tot} length-power
points, {n_ok} have a nonempty quality-certified domain and {n_cert} of those
also meet `Y_min`; the smallest optimized `Y_lb` among them is {y_cert_min:.2e}.
At {n_unc_ub} of the {n_tot - n_ok} uncertified points the upper bound
`max s^7` still exceeds `Y_min`, so the bounds leave those points unresolved
rather than excluding them.  The fidelity-feasibility boundary is resolved to
{fmt(bnd_res, 2)} in `C`.  Requiring tuning margin moves the selected wavelength
and changes the optimized yield by the amounts in Supplementary Table S2;
{len(failing)} enumerated scenarios empty the certified domain at 5 km /
-10 dBm/ch, and the intersection of the certified domains over all scenarios is
{"empty" if rob_empty else "nonempty"}, so no single wavelength is certified
across the whole enumerated set.

**What these results do not establish.**  They do not show that the physical
assumptions hold in deployed fibre; they do not make any accepted logical state
known to be correct; `T^7` is not an upper bound once background replacement is
included; an empty certified set does not prove physical impossibility; the
lower-bound optimum need not be the optimum of exact yield; and no quantity
reported here is a secret-key rate.
"""
    open(os.path.join(RES, "results_evidence_outline.md"), "w",
         encoding="utf-8").write(ev)
    print("  wrote results_evidence_outline.md")

    # --------------------------------------------------------- run metadata --
    meta = {"generated_utc": io.utcnow(),
            "software": io.package_versions(),
            "source_sha256": io.source_hashes(),
            "reproduction_command": "python scripts/run_all.py full",
            "configs": {}, "seeds": {}, "numerical_settings": {},
            "manifests": [], "outputs": {}}
    for name in ("illustrative_baseline.yaml", "validation.yaml",
                 "legacy_reserved_window.yaml"):
        p = os.path.join(ROOT, "configs", name)
        if os.path.exists(p):
            meta["configs"][name] = {"sha256": io.file_hash(p),
                                     "content": cf.load(name)}
    vcfg = cf.load("validation.yaml")
    meta["seeds"] = {
        "decoder_random_key_seed_base": vcfg["decoder_sweep"]["key_seed"],
        "n_random_key_sets": vcfg["decoder_sweep"]["n_random_key_sets"],
        "key_generation_rule": ("common axis per location drawn from {x,y,z}; "
                                "both angles uniform on [0, 2 pi); NOT Haar "
                                "sampling of all admissible commuting pairs"),
        "optical_dilation_random_seed": vcfg["optical_dilation"]["random_seed"]}
    meta["numerical_settings"] = {
        "gauss_hermite_order": cfg["numerics"]["gauss_hermite_order"],
        "gauss_hermite_check_order": cfg["numerics"]["gauss_hermite_check_order"],
        "grid_step_ghz": cfg["search"]["grid_step_ghz"],
        "refine_step_ghz": cfg["search"]["refine_step_ghz"],
        "guard_band_ghz": guard, "tuning_margin_ghz": margin,
        "thermal_tail_tolerances": vcfg["optical_dilation"]["thermal_tail_tolerances"],
        "boundary_residual_target_in_C": 1e-6,
        "convergence_targets": {"relative_Y_lb": 1e-3, "absolute_C": 1e-6}}
    for p in sorted(glob.glob(os.path.join(RES, "*", "run_manifest_*.json"))):
        with open(p) as fh:
            m = json.load(fh)
        meta["manifests"].append({
            "path": os.path.relpath(p, ROOT), "run_id": m.get("run_id"),
            "stage": m.get("stage"), "finished_utc": m.get("finished_utc"),
            "wall_time_s": m.get("wall_time_s"),
            "tolerances": m.get("tolerances", {}),
            "n_outputs": len(m.get("outputs", []))})
    meta["outputs"] = {
        "figures": sorted(os.path.relpath(x, ROOT) for x in
                          glob.glob(os.path.join(RES, "figures", "*.pdf"))),
        "tables": sorted(os.path.relpath(x, ROOT) for x in
                         glob.glob(os.path.join(RES, "tables", "*.csv"))),
        "data_csv_count": len(glob.glob(os.path.join(RES, "*", "*.csv")))}
    io.write_json(os.path.join(RES, "run_metadata.json"), meta)
    print("  wrote run_metadata.json")


if __name__ == "__main__":
    print("building reports from stored results ...")
    main()
