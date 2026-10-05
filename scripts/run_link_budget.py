#!/usr/bin/env python
"""Operating points at the classical link-budget launch power, and the
base-error sweep.

Two questions are answered here.

1. When every classical channel is launched at the smallest power that still
   closes its data link, which quantum wavelength is selected and is it
   certified?  The minimum launch power follows from the receiver sensitivity,
   the span loss, the multiplexing optics and a margin (src/link_budget.py),
   for each receiver declared in the configuration.

2. How large may the per-traversal base error e0 be before certification is
   lost?  The optical components are computed once per operating point; e0
   only enters through the traversal identity weights, so the sweep over the
   literature anchors and the bisection for the largest admissible e0 reuse the
   same optical solution.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import config as cf
import link_budget as lb
import local_protocol as lp
import optical_channel as oc
import optical_link as ol
import protocol as pr
import reliability as rl
import runio as io
from units import hz_to_nm

N, TT = 7, 1
PREDECLARED = [(2.0, -15.0), (3.0, -13.0), (5.0, -10.0), (10.0, -5.0)]


def optical_solution(cfg, f_allowed, L, P):
    """Per-candidate (eta, nu, sigma^2) at one operating point."""
    sc = cf.build_scenario(cfg, length_km=L, launch_dbm=P)
    n = len(f_allowed)
    eta = np.zeros((n, 3)); nu = np.zeros((n, 3)); sg = np.zeros((n, 3))
    for i, f in enumerate(f_allowed):
        r = ol.evaluate_candidate(sc, f, want_components=False)
        eta[i] = r["eta"]; nu[i] = r["nu"]; sg[i] = r["sigma2_phi"]
    return eta, nu, sg


def outcome(eta, nu, sg, e0, lam, F_min, Y_min):
    """Certificate, selection and yield for one base-error value."""
    T, nuf, _ = oc.compose_passes(eta, nu)
    res = oc.accepted_optical(T, nuf)
    z = lp.z_from_sigma2(sg)
    b = np.stack([lp.traversal_weights(e0, z[..., m])[..., 0] for m in range(3)],
                 axis=-1)
    q = lp.affected_probability(res.wI, b)
    C = rl.correctable_weight(q, N, TT)
    Ylb = res.s ** N * C
    mF = (res.s > 0) & (C >= F_min)
    out = dict(n_Omega_F=int(mF.sum()), max_C=float(np.max(C)))
    if mF.any():
        k = int(np.nonzero(mF)[0][np.argmax(Ylb[mF])])
        out.update(lambda_star_nm=float(lam[k]), s=float(res.s[k]),
                   q=float(q[k]), C=float(C[k]), Y_lb=float(Ylb[k]),
                   Y_ub=float(res.s[k] ** N), k=k,
                   status=("certified for F_min and Y_min" if Ylb[k] >= Y_min
                           else "quality certified; yield NOT certified"))
    else:
        out.update(lambda_star_nm=np.nan, s=np.nan, q=np.nan, C=np.nan,
                   Y_lb=np.nan, Y_ub=float(np.max(res.s) ** N), k=None,
                   status="NOT certified: Omega_F empty")
    return out


def main():
    t0 = time.time()
    cfg = cf.load("illustrative_baseline.yaml")
    run_id = os.environ.get("QCFC_RUN_ID", "illustrative_baseline")
    d = io.run_dir(run_id)
    man = io.Manifest(run_id, "link-budget operating points and base-error sweep", cfg)
    tg = cfg["targets"]
    F_min, Y_min = tg["F_min"], tg["Y_min"]
    e0_nom = cfg["link"]["e0_per_pass"]
    q_star = pr.certificate_threshold_q(F_min)

    dep = cf.build_deployment(cfg)
    grid = cf.build_grid(cfg, dep)
    _, _, _, allowed, _ = cf.build_masks(cfg, grid, dep)
    f_allowed = grid[allowed]
    lam = hz_to_nm(f_allowed)

    atten = cf.build_attenuation(cfg)
    alpha_c = lb.worst_classical_attenuation_db_km(atten, hz_to_nm(dep.freqs_hz))
    mux, demux, margin = lb.budget_terms(cfg)

    # operating points: the predeclared four, then one per receiver and length
    points = [dict(kind="predeclared", receiver="-", L=L, P=P) for L, P in PREDECLARED]
    for rx in lb.receivers(cfg):
        for L in cfg["classical_link"]["evaluation_lengths_km"]:
            P = float(lb.minimum_launch_dbm(L, rx["rx_sensitivity_dbm"], alpha_c,
                                            mux, demux, margin))
            points.append(dict(kind="link-budget minimum", receiver=rx["id"],
                               receiver_label=rx["label"],
                               rx_sensitivity_dbm=rx["rx_sensitivity_dbm"],
                               receiver_source=rx["source"], L=float(L), P=P))

    anchors = cfg["base_error_sweep"]["anchors"]
    keys_id = pr.key_set("identity")
    lb_rows, sweep_rows = [], []
    for pt in points:
        eta, nu, sg = optical_solution(cfg, f_allowed, pt["L"], pt["P"])
        e0m_all = pr.e0_max_for_certification(eta, nu, sg, q_star)
        if np.isfinite(e0m_all).any():
            j = int(np.nanargmax(e0m_all))
            e0_max, lam_e0 = float(e0m_all[j]), float(lam[j])
        else:
            e0_max, lam_e0 = np.nan, np.nan
        nom = outcome(eta, nu, sg, e0_nom, lam, F_min, Y_min)
        fe = np.nan
        if nom["k"] is not None:
            k = nom["k"]
            cp = pr.channel_point(eta[k], nu[k], e0_nom, lp.z_from_sigma2(sg[k]))
            fe, _ = pr.exact_fidelity(cp, keys_id, "pauli_enumeration")
        base = dict(point_kind=pt["kind"], receiver=pt["receiver"],
                    length_km=pt["L"], launch_dbm=round(pt["P"], 4))
        if pt["kind"] != "predeclared":
            base.update(rx_sensitivity_dbm=pt["rx_sensitivity_dbm"],
                        worst_classical_alpha_db_km=alpha_c,
                        mux_db=mux, demux_db=demux, margin_db=margin)
        lb_rows.append(dict(base, e0=e0_nom, n_Omega_F=nom["n_Omega_F"],
                            max_C_over_Omega_q=nom["max_C"],
                            lambda_star_nm=nom["lambda_star_nm"], s=nom["s"],
                            q=nom["q"], C=nom["C"], Fe_exact_identity=fe,
                            Y_lb=nom["Y_lb"], Y_ub=nom["Y_ub"], status=nom["status"],
                            e0_max_for_certification=e0_max,
                            e0_max_wavelength_nm=lam_e0,
                            source=pt.get("receiver_source", "predeclared point")))
        for a in anchors:
            o = outcome(eta, nu, sg, float(a["e0"]), lam, F_min, Y_min)
            sweep_rows.append(dict(base, e0=float(a["e0"]), e0_source=a["source"],
                                   n_Omega_F=o["n_Omega_F"],
                                   max_C_over_Omega_q=o["max_C"],
                                   lambda_star_nm=o["lambda_star_nm"], C=o["C"],
                                   Y_lb=o["Y_lb"], status=o["status"],
                                   e0_max_for_certification=e0_max))
        print(f"  {pt['kind']:20s} {pt['receiver']:18s} L={pt['L']:5.1f} km "
              f"P={pt['P']:+7.2f} dBm  nOmF={nom['n_Omega_F']:4d}  "
              f"lam*={nom['lambda_star_nm']:9.3f}  Y_lb={nom['Y_lb']:.3e}  "
              f"e0_max={100*e0_max if np.isfinite(e0_max) else float('nan'):.3f}%  "
              f"({time.time()-t0:.0f} s)", flush=True)

    # the same quantities on the map's length grid, for plotting along the
    # link-budget line of each receiver
    mp = cfg["map"]
    curve_L = np.arange(mp["length_km_min"], mp["length_km_max"] + 1e-9,
                        mp["length_km_step"])
    curve_rows = []
    for rx in lb.receivers(cfg):
        for L in curve_L:
            P = float(lb.minimum_launch_dbm(L, rx["rx_sensitivity_dbm"], alpha_c,
                                            mux, demux, margin))
            eta, nu, sg = optical_solution(cfg, f_allowed, float(L), P)
            e0m_all = pr.e0_max_for_certification(eta, nu, sg, q_star)
            nom = outcome(eta, nu, sg, e0_nom, lam, F_min, Y_min)
            ok = np.isfinite(e0m_all).any()
            curve_rows.append(dict(
                receiver=rx["id"], data_rate_gbps=rx["data_rate_gbps"],
                rx_sensitivity_dbm=rx["rx_sensitivity_dbm"], length_km=float(L),
                launch_dbm=round(P, 4), e0=e0_nom, n_Omega_F=nom["n_Omega_F"],
                max_C_over_Omega_q=nom["max_C"],
                lambda_star_nm=nom["lambda_star_nm"], Y_lb=nom["Y_lb"],
                Y_ub=nom["Y_ub"], status=nom["status"],
                e0_max_for_certification=(float(np.nanmax(e0m_all)) if ok else np.nan),
                e0_max_wavelength_nm=(float(lam[int(np.nanargmax(e0m_all))])
                                      if ok else np.nan)))
        print(f"  link-budget curve {rx['id']}: {len(curve_L)} lengths "
              f"({time.time()-t0:.0f} s)", flush=True)

    man.add_output(io.write_csv(os.path.join(d, "link_budget_points.csv"), lb_rows))
    man.add_output(io.write_csv(os.path.join(d, "link_budget_curve.csv"), curve_rows))
    man.add_output(io.write_csv(os.path.join(d, "base_error_sweep.csv"), sweep_rows))
    man.tolerance("certificate_threshold_q_star", q_star)
    man.note("Launch power at the link-budget minimum for each declared receiver; "
             "e0 swept over published anchors with the optical solution fixed.")
    man.save(os.path.join(d, "run_manifest_link_budget.json"))
    print(f"\nlink budget and base-error sweep: {len(lb_rows)} points "
          f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
