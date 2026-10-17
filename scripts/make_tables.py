#!/usr/bin/env python
"""Manuscript tables (Methods: tables), generated from stored run
outputs only.

  Main Table 1  model inputs and numerical settings, read from the executed
                configuration
  Main Table 2  wavelength-selection outcomes at the predeclared operating
                points, for every selection rule
  Table S1      numerical verification: asserted checks, then report-only
                diagnostics, kept in separate parts
  Table S2      sensitivity and convergence
  Table S3      operating points at the classical link-budget minimum and the
                base-error requirement

Every table is written as CSV, Markdown and LaTeX from the same data frame.
Supporting CSVs (FWM power accounting, mode comparison, correlated-fault
control, selector comparison, certified regions, migration record) are copied
into results/tables/supporting/ without being promoted to manuscript tables.
"""
import glob
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import runio as io
import config as cf

RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
SUP = os.path.join(TAB, "supporting")
os.makedirs(TAB, exist_ok=True)
os.makedirs(SUP, exist_ok=True)

# run ids of the predeclared operating points, in the order used in the manuscript
POINTS = [("op_2km_m15dBm", 2.0, -15.0, 1.0, "interior example"),
          ("op_3km_m13dBm", 3.0, -13.0, 1.0, "interior example"),
          ("op_5km_m10dBm", 5.0, -10.0, 1.0, "original comparison point"),
          ("op_6km_m10dBm", 6.0, -10.0, 1.0, "active fidelity constraint comparison"),
          ("op_10km_m5dBm", 10.0, -5.0, 1.0, "legacy comparison point"),
          ("op_2km_m15dBm_margin0", 2.0, -15.0, 0.0, "interior, zero added margin"),
          ("op_3km_m13dBm_margin0", 3.0, -13.0, 0.0, "interior, zero added margin"),
          ("op_5km_m10dBm_margin0", 5.0, -10.0, 0.0,
           "original comparison, zero added margin"),
          ("op_10km_m5dBm_margin0", 10.0, -5.0, 0.0,
           "legacy comparison, zero added margin")]

# Keep the original four diagnostic anchors when adding the Table 2 example.
DIAGNOSTIC_POINTS = [p for p in POINTS if p[3] == 1.0 and p[1] != 6.0]


def emit(name, df, caption):
    df = df.copy()
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    with open(os.path.join(TAB, f"{name}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"**{caption}**\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n")
    with open(os.path.join(TAB, f"{name}.tex"), "w", encoding="utf-8") as fh:
        fh.write(to_latex(df, caption, f"tab:{name}"))
    print(f"  wrote {name}.csv/.md/.tex  ({len(df)} rows)")


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _esc(x):
    """Escape one cell for LaTeX text mode."""
    s = "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    return "".join(_TEX.get(ch, ch) for ch in s)


def to_latex(df, caption, label):
    """Dependency-free longtable export (pandas' writer needs jinja2)."""
    cols = list(df.columns)
    spec = "".join("p{%.2f\\linewidth}" % (0.95 / max(len(cols), 1))
                   for _ in cols)
    head = " & ".join(r"\textbf{" + _esc(c) + "}" for c in cols) + r" \\"
    lines = [r"\begin{longtable}{" + spec + "}",
             r"\caption{" + _esc(caption) + r"}\label{" + label + r"}\\",
             r"\hline", head, r"\hline", r"\endfirsthead",
             r"\hline", head, r"\hline", r"\endhead"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_esc(row[c]) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{longtable}", ""]
    return "\n".join(lines)


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


def sig(x, n=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(v):
        return "undefined"
    return f"{v:.{n}g}"


def main():
    cfg = cf.load("illustrative_baseline.yaml")

    # ------------------------------------------------------------ Main Table 1
    t1 = pd.DataFrame(io.provenance_rows(cfg))
    t1 = t1.rename(columns={"dataset": "parameter", "value": "value_or_setting",
                            "reference_plane_or_traversal": "reference_plane"})
    t1 = t1[["parameter", "symbol", "value_or_setting", "unit", "reference_plane",
             "provenance", "sensitivity_range", "note"]] if "provenance" in t1 \
        else t1[["parameter", "symbol", "value_or_setting", "unit",
                 "reference_plane", "source", "provenance_category",
                 "sensitivity_range", "note"]]
    emit("main_table1_model_inputs_and_numerical_settings", t1,
         "Main Table 1 | Model inputs and numerical settings, generated from the "
         "executed configuration. The value, its provenance and the range used "
         "for sensitivity are separate columns. No row is a measurement: no "
         "measured fibre spectrum, receiver characterization or deployment "
         "record was supplied to this study.")

    # ------------------------------------------------------------ Main Table 2
    rows = []
    for rid, L, P, margin, role in POINTS:
        sel = rd(rid, "selection_summary.csv")
        if sel is None:
            continue
        man_path = os.path.join(RES, rid, "run_manifest_baseline.json")
        guard = cfg["search"]["guard_band_ghz"]
        if os.path.exists(man_path):
            with open(man_path) as fh:
                mcfg = json.load(fh)["config"]
            guard = mcfg["search"]["guard_band_ghz"]
            margin = mcfg["search"].get("tuning_margin_ghz", margin)
        exact = rd(rid, "exact_decoder_points.csv")
        for _, r in sel.iterrows():
            lam = r.get("lambda_selected_nm")
            fe_key = "identity keys, Pauli enumeration"
            fe = r.get("Fe_exact")
            if isinstance(fe, str) or pd.isna(fe):
                fe, fe_key = "not evaluated", "-"
            rows.append(dict(
                operating_point=f"{L:g} km, {P:+.0f} dBm/ch",
                role=role, length_km=L, launch_dbm=P,
                guard_ghz=guard, added_tuning_margin_ghz=margin,
                selection_rule=r["selection_rule"],
                search_domain=r.get("searched_over", "-"),
                lambda_selected_nm=(sig(lam, 8) if not pd.isna(lam) else "undefined"),
                s=sig(r.get("s")), q=sig(r.get("q")), C_bound=sig(r.get("C_bound"), 8),
                Fe_exact=(sig(fe, 8) if not isinstance(fe, str) else fe),
                exact_fidelity_key_case=fe_key,
                Y_lb=sig(r.get("Y_lb")), Y_ub=sig(r.get("Y_ub")),
                Y_exact=(sig(r.get("Y_exact")) if not isinstance(r.get("Y_exact"), str)
                         else r.get("Y_exact")),
                certificate_status=r.get("certificate_status", "")))
    t2_all = pd.DataFrame(rows)
    # The main table carries the main practical illustration (guard +
    # 1 GHz added margin); the zero-margin comparison rows go to a supporting CSV
    # and into the tuning-margin part of Table S2.
    t2 = t2_all[t2_all["added_tuning_margin_ghz"] == 1.0].reset_index(drop=True)
    t2_all[t2_all["added_tuning_margin_ghz"] == 0.0].to_csv(
        os.path.join(SUP, "main_table2_zero_margin_comparison.csv"), index=False)
    emit("main_table2_wavelength_selection_outcomes", t2,
         "Main Table 2 | Wavelength-selection outcomes at the predeclared "
         "operating points. Every rule searches the domain named in "
         "search_domain. `undefined` marks an empty constrained set or a rule "
         "whose wavelength is not an allowed candidate. Y_exact uses the stated "
         "key case; C_bound is the lower-bound certificate and is key "
         "independent.")

    # --------------------------------------------------------- Table S1
    parts = []
    for rid, fname, stage in (("validation", "validation_table_s2_optics.csv",
                               "B optical"),
                              ("validation", "validation_table_s2_decoder.csv",
                               "A+C logical")):
        df = rd(rid, fname)
        if df is None:
            continue
        df = df.copy()
        df.insert(0, "stage", stage)
        parts.append(df)
    s1 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(s1):
        if "row_kind" not in s1:
            s1["row_kind"] = "assert"
        s1["row_kind"] = s1["row_kind"].fillna("assert")
        s1 = pd.concat([s1[s1["row_kind"] == "assert"],
                        s1[s1["row_kind"] != "assert"]], ignore_index=True)
        keep = ["stage", "row_kind", "validation_case", "parameters_or_config_id",
                "independent_method", "reference_quantity", "computed_quantity",
                "absolute_error", "relative_error_if_defined", "tolerance",
                "cutoff_or_grid", "pass_fail_and_reason"]
        s1 = s1.reindex(columns=[c for c in keep if c in s1.columns])
    n_assert = int((s1["row_kind"] == "assert").sum()) if len(s1) else 0
    n_diag = len(s1) - n_assert
    n_fail = int(s1["pass_fail_and_reason"].astype(str).str.startswith("FAIL").sum()) \
        if len(s1) else 0
    emit("supp_table_s1_numerical_verification", s1,
         f"Supplementary Table S1 | Numerical verification. {n_assert} asserted "
         f"checks ({n_fail} failures) are listed first, followed by {n_diag} "
         f"report-only diagnostics that carry no tolerance and are not counted "
         f"as checks. Tolerances are relative wherever the reference is nonzero; "
         f"a dimensionful absolute tolerance is never applied to a nonzero "
         f"reference.")

    # --------------------------------------------------------- Table S2
    s2 = []
    scn = rd("illustrative_baseline", "scenario_sensitivity.csv")
    rob = rd("illustrative_baseline", "robustness_summary.csv")
    if scn is not None:
        fixed = {}
        if rob is not None:
            fx = rob[rob["mode"] == "fixed nominal wavelength"]
            for _, r in fx.iterrows():
                fixed[(r["anchor"], r["scenario"])] = r
        for _, r in scn.iterrows():
            f = fixed.get((r["anchor"], r["label"]))
            s2.append(dict(
                part="scenario sensitivity", case=r["label"], anchor=r["anchor"],
                allowed_domain=("Omega_q at the configured guard and margin"),
                M_theta_max_C_over_Omega_q=sig(r.get("max_C_over_Omega_q"), 8),
                fixed_wavelength_certificate=(sig(f["C_bound"], 8) if f is not None
                                              else "-"),
                fixed_wavelength_certified=(bool(f["certified"]) if f is not None
                                            else "-"),
                selected_lambda_nm=sig(r.get("lambda_nm"), 8),
                max_Y_lb=sig(r.get("Y_lb")),
                relative_yield_change="reference" if r["label"] == "nominal" else "",
                refinement="-",
                certification_status=r.get("target_status", ""),
                interpretation=r.get("uncertainty_or_sensitivity_interpretation", "")))
        # relative yield change against the nominal scenario at the same anchor
        base = {r["anchor"]: r for _, r in scn.iterrows() if r["label"] == "nominal"}
        for row, (_, r) in zip([x for x in s2 if x["part"] == "scenario sensitivity"],
                               scn.iterrows()):
            b = base.get(r["anchor"])
            if (b is not None and np.isfinite(float(r.get("Y_lb", np.nan)))
                    and np.isfinite(float(b.get("Y_lb", np.nan)))
                    and float(b["Y_lb"]) > 0 and r["label"] != "nominal"):
                row["relative_yield_change"] = sig(
                    float(r["Y_lb"]) / float(b["Y_lb"]) - 1.0, 4)
            elif r["label"] != "nominal":
                row["relative_yield_change"] = "undefined"
    gm = rd("illustrative_baseline", "guard_margin_study.csv")
    if gm is not None:
        for _, r in gm[gm["guard_ghz"] == 50.0].iterrows():
            s2.append(dict(
                part="tuning margin", case=f"margin {r['tuning_margin_ghz']:g} GHz",
                anchor=f"{r['length_km']:g} km, {r['launch_dbm']:+.0f} dBm/ch",
                allowed_domain=(f"guard {r['guard_ghz']:g} GHz + margin "
                                f"{r['tuning_margin_ghz']:g} GHz"),
                M_theta_max_C_over_Omega_q="-",
                fixed_wavelength_certificate="-", fixed_wavelength_certified="-",
                selected_lambda_nm=sig(r["lambda_star_nm"], 8),
                max_Y_lb=sig(r["Y_lb_at_star"]),
                relative_yield_change=(sig(r["relative_Y_lb_change_vs_zero_margin"], 4)
                                       if r.get("both_optima_exist", False)
                                       else "undefined: both optima do not exist"),
                refinement="-", certification_status=r["status"],
                interpretation="margin is a scenario choice, not a measured "
                               "hardware tolerance"))
    pw = rd("illustrative_baseline", "fwm_profile_width.csv")
    if pw is not None:
        for _, r in pw.iterrows():
            s2.append(dict(
                part="FWM spectral profile",
                case=f"{r['fwm_profile_rms_ghz']:g} GHz rms",
                anchor=f"{r['length_km']:g} km, {r['launch_dbm']:+.0f} dBm/ch",
                allowed_domain="Omega_q at the configured guard and margin",
                M_theta_max_C_over_Omega_q=sig(r["max_C_over_allowed"], 8),
                fixed_wavelength_certificate="-", fixed_wavelength_certified="-",
                selected_lambda_nm=sig(r["lambda_star_nm"], 8),
                max_Y_lb=sig(r["Y_lb_at_star"]),
                relative_yield_change=sig(r["relative_Y_lb_change_vs_delta_line"], 4),
                refinement="-", certification_status=r["status"],
                interpretation=r["profile_status"]))
    for rid, L, P, margin, role in DIAGNOSTIC_POINTS:
        cv = rd(rid, "convergence.csv")
        if cv is None:
            continue
        for _, r in cv.iterrows():
            s2.append(dict(
                part="convergence", case=f"{r['refinement']}: {r['setting']}",
                anchor=f"{L:g} km, {P:+.0f} dBm/ch",
                allowed_domain=f"held fixed: {r['held_fixed']}",
                M_theta_max_C_over_Omega_q="-", fixed_wavelength_certificate="-",
                fixed_wavelength_certified="-",
                selected_lambda_nm=sig(r["lambda_star_nm"], 8),
                max_Y_lb=sig(r["max_Y_lb"]),
                relative_yield_change=sig(r["relative_change_Y_lb"], 4),
                refinement=(f"|dC| = {sig(r['absolute_change_C'], 3)}, "
                            f"|dlambda| = {sig(r['lambda_shift_nm'], 3)} nm"),
                certification_status=f"n_Omega_F = {r['n_Omega_F']}",
                interpretation="targets: <1e-3 relative in Y_lb and <1e-6 "
                               "absolute in C"))
    for rid, L, P, margin, role in DIAGNOSTIC_POINTS:
        mr = rd(rid, "mode_reference.csv")
        if mr is None:
            continue
        for _, r in mr.iterrows():
            s2.append(dict(
                part="receiver model", case=f"{r['candidate_role']} candidate",
                anchor=f"{L:g} km, {P:+.0f} dBm/ch",
                allowed_domain=f"lambda = {r['wavelength_nm']:.4f} nm",
                M_theta_max_C_over_Omega_q="-",
                fixed_wavelength_certificate=sig(r["C_scalar"], 8),
                fixed_wavelength_certified="-",
                selected_lambda_nm=sig(r["wavelength_nm"], 8),
                max_Y_lb=sig(r["Y_lb_scalar"]),
                relative_yield_change=sig(r["Y_lb_relative_difference"], 4),
                refinement=(f"dT/T = {sig(r['T_relative_difference'], 3)}, "
                            f"dnu/nu = {sig(r['nu_relative_difference'], 3)}, "
                            f"dC = {sig(r['C_absolute_difference'], 3)}"),
                certification_status="full-chain spectral reference vs the scalar "
                                     "recurrence",
                interpretation="the scalar recurrence assumes the declared mode "
                               "restoration at every traversal output; the "
                               "reference propagates the field through the whole "
                               "chain and projects once"))
    emit("supp_table_s2_sensitivity_and_convergence", pd.DataFrame(s2),
         "Supplementary Table S2 | Sensitivity and convergence. Scenario ranges "
         "are declared sensitivity choices, not statistical confidence "
         "intervals. M_theta is the maximum certificate over the allowed domain: "
         "M_theta < F_min means no candidate is certified by this bound, not "
         "that exact fidelity fails. Relative yield changes are quoted against "
         "the reference rule named in each part.")

    # --------------------------------------------------------- Table S3
    lbp = rd("illustrative_baseline", "link_budget_points.csv")
    bes = rd("illustrative_baseline", "base_error_sweep.csv")
    if lbp is not None:
        rx_lab = {r["id"]: r["label"]
                  for r in cfg.get("classical_link", {}).get("receivers", [])}
        s3 = []
        for _, r in lbp.iterrows():
            pred = r["point_kind"] == "predeclared"
            s3.append(dict(
                part="link budget" if not pred else "predeclared point",
                receiver=("-" if pred else rx_lab.get(r["receiver"], r["receiver"])),
                length_km=sig(r["length_km"], 4),
                launch_dbm=sig(r["launch_dbm"], 5),
                e0=sig(r["e0"], 4), n_Omega_F=int(r["n_Omega_F"]),
                lambda_selected_nm=(sig(r["lambda_star_nm"], 8)
                                    if np.isfinite(r["lambda_star_nm"]) else "undefined"),
                C_bound=sig(r["C"], 8) if np.isfinite(r["C"]) else "undefined",
                Y_lb=sig(r["Y_lb"]) if np.isfinite(r["Y_lb"]) else "undefined",
                largest_admissible_e0=(sig(r["e0_max_for_certification"], 4)
                                       if np.isfinite(r["e0_max_for_certification"])
                                       else "none"),
                certificate_status=r["status"], source=r["source"]))
        if bes is not None:
            for _, r in bes.iterrows():
                pred = r["point_kind"] == "predeclared"
                s3.append(dict(
                    part="base-error anchor",
                    receiver=("-" if pred else rx_lab.get(r["receiver"], r["receiver"])),
                    length_km=sig(r["length_km"], 4),
                    launch_dbm=sig(r["launch_dbm"], 5),
                    e0=sig(r["e0"], 4), n_Omega_F=int(r["n_Omega_F"]),
                    lambda_selected_nm=(sig(r["lambda_star_nm"], 8)
                                        if np.isfinite(r["lambda_star_nm"])
                                        else "undefined"),
                    C_bound=sig(r["C"], 8) if np.isfinite(r["C"]) else "undefined",
                    Y_lb=sig(r["Y_lb"]) if np.isfinite(r["Y_lb"]) else "undefined",
                    largest_admissible_e0=(sig(r["e0_max_for_certification"], 4)
                                           if np.isfinite(r["e0_max_for_certification"])
                                           else "none"),
                    certificate_status=r["status"], source=r["e0_source"]))
        emit("supp_table_s3_link_budget_and_base_error", pd.DataFrame(s3),
             "Supplementary Table S3 | Operating points at the classical "
             "link-budget minimum and the base-error requirement. The launch "
             "power is the smallest that closes the classical link for the named "
             "receiver (sensitivity, worst classical attenuation, multiplexer, "
             "demultiplexer and margin). The largest admissible e0 is the "
             "largest per-traversal base error at which some allowed candidate "
             "still meets C >= F_min; the anchor rows re-evaluate each point at "
             "the published e0 values with the optical solution unchanged.")

    # ------------------------------------------------- supporting stored CSVs
    keep = [("illustrative_baseline", "fwm_slot_split.csv"),
            ("illustrative_baseline", "guard_margin_study.csv"),
            ("illustrative_baseline", "guard_geometry.csv"),
            ("illustrative_baseline", "certification_boundary.csv"),
            ("illustrative_baseline", "operating_map.csv"),
            ("illustrative_baseline", "ablations.csv"),
            ("illustrative_baseline", "robustness_summary.csv"),
            ("illustrative_baseline", "selector_stability.csv"),
            ("illustrative_baseline", "ensemble_averaging.csv"),
            ("illustrative_baseline", "fwm_profile_width.csv"),
            ("illustrative_baseline", "link_budget_curve.csv"),
            ("validation", "correlated_fault_control.csv"),
            ("validation", "migration_record.csv"),
            ("validation", "optical_validation.csv"),
            ("validation", "optical_convergence.csv"),
            ("validation", "attenuation_provenance.csv")]
    for rid, fname in keep:
        p = os.path.join(RES, rid, fname)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(SUP, f"{rid}__{fname}"))
    for rid, L, P, margin, role in POINTS:
        for fname in ("selection_summary.csv", "certified_regions.csv",
                      "mode_reference.csv", "convergence.csv",
                      "exact_decoder_points.csv"):
            p = os.path.join(RES, rid, fname)
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join(SUP, f"{rid}__{fname}"))
    print(f"  copied {len(glob.glob(os.path.join(SUP, '*.csv')))} supporting CSVs")


if __name__ == "__main__":
    print("building tables from stored results ...")
    main()
