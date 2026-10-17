#!/usr/bin/env python
"""Reproduce the study.

  python scripts/run_all.py quick    # Stages A-C only (~25 s)
  python scripts/run_all.py full     # Stages A-G, figures, tables, reports
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

QUICK = [("Stage A+C  decoder validation", ["scripts/validate_decoder.py"], {}),
         ("Stage B    optical validation", ["scripts/validate_optics.py"], {}),
         ("tests", ["-m", "pytest", "tests", "-q"], {})]

def _pt(L, P, margin=1.0, tag=None):
    rid = tag or f"op_{L:g}km_{P:+.0f}dBm".replace("+", "p").replace("-", "m")
    return (f"Stage D    operating point {L:g} km, {P:+.0f} dBm/ch, "
            f"margin {margin:g} GHz",
            ["scripts/run_baseline.py"],
            {"QCFC_RUN_ID": rid, "QCFC_LENGTH_KM": str(L),
             "QCFC_LAUNCH_DBM": str(P), "QCFC_MARGIN_GHZ": str(margin)})


# The four PREDECLARED operating points of the Methods, all with
# the 50 GHz carrier guard and the illustrative 1 GHz added tuning margin, plus
# the same points at zero added margin for comparison. The 6 km, -10 dBm
# point supplies the active-fidelity-constraint comparison in Main Table 2.
FULL = QUICK + [
    _pt(2.0, -15.0), _pt(3.0, -13.0), _pt(5.0, -10.0),
    _pt(6.0, -10.0), _pt(10.0, -5.0),
    _pt(2.0, -15.0, 0.0, "op_2km_m15dBm_margin0"),
    _pt(3.0, -13.0, 0.0, "op_3km_m13dBm_margin0"),
    _pt(5.0, -10.0, 0.0, "op_5km_m10dBm_margin0"),
    _pt(10.0, -5.0, 0.0, "op_10km_m5dBm_margin0"),
    ("Stage D    legacy reserved-window comparison", ["scripts/run_baseline.py"],
     {"QCFC_BASELINE_CONFIG": "legacy_reserved_window.yaml",
      "QCFC_RUN_ID": "legacy_reserved_window"}),
    ("Stage E    operating map", ["scripts/run_operating_map.py"], {}),
    ("Stage E    guard geometry and tuning margin", ["scripts/run_guard_margin.py"], {}),
    ("Stage E    certification boundary refinement", ["scripts/run_boundary_refine.py"], {}),
    ("Stage E    FWM spectral-profile diagnostic", ["scripts/run_profile_width.py"], {}),
    ("Stage E    link-budget points and base-error sweep", ["scripts/run_link_budget.py"], {}),
    ("Stage F    sensitivity and ablations", ["scripts/run_sensitivity.py"], {}),
    ("figures", ["scripts/make_figures.py"], {}),
    ("tables", ["scripts/make_tables.py"], {}),
    ("reports", ["scripts/make_reports.py"], {}),
]


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "quick").lower()
    steps = QUICK if mode == "quick" else FULL
    t0 = time.time()
    for name, args, env in steps:
        print(f"\n=== {name} " + "=" * max(0, 56 - len(name)))
        e = dict(os.environ); e.update(env)
        r = subprocess.run([PY] + args, cwd=ROOT, env=e)
        if r.returncode != 0:
            sys.exit(f"FAILED: {name}")
    print(f"\nAll steps completed in {time.time()-t0:.0f} s. "
          f"Outputs are under results/.")


if __name__ == "__main__":
    main()
