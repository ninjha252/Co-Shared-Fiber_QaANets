#!/usr/bin/env python
"""Stages A and C: synthetic benchmark, Steane construction, recovery
verification, and the decoder noise sweep (Methods: encoder, recovery and noise sweep)."""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import config as cf
import local_protocol as lp
import optical_channel as oc
import protocol as pr
import reliability as rl
import runio as io
import steane as st

RUN_ID = os.environ.get("QCFC_RUN_ID", "validation")
TOL_BENCH = 1e-12
TOL_EXACT = 1e-10


def main():
    cfg = cf.load("validation.yaml")
    d = io.run_dir(RUN_ID)
    man = io.Manifest(RUN_ID, "A+C decoder validation", cfg)
    man.tolerance("benchmark_relative", TOL_BENCH)
    man.tolerance("exact_identity_absolute", TOL_EXACT)
    rows_s2 = []          # Supplementary Table S2 rows

    def record(case, params, method, ref, got, cutoff="", tail="", note="",
               rel=None, kind="assert"):
        """Every quantity recorded here is a DIMENSIONLESS residual of order one
        (a fidelity, a probability weight, a matrix-element difference), so an
        absolute tolerance is the meaningful criterion; pass `rel` where a
        relative comparison against a nonzero reference is wanted instead."""
        ae = abs(got - ref) if isinstance(ref, float) else np.nan
        re_ = ae / abs(ref) if isinstance(ref, float) and ref != 0 else np.nan
        if rel is not None:
            ok = bool(np.isfinite(re_) and re_ <= rel)
            tolerance = f"rel<={rel:g}"
        elif isinstance(ref, float):
            ok = ae <= TOL_EXACT
            tolerance = f"abs<={TOL_EXACT:g} (dimensionless residual)"
        else:
            ok = bool(got)
            tolerance = "boolean check"
        rows_s2.append(dict(validation_case=case, parameters_or_config_id=params,
                            independent_method=method, reference_quantity=ref,
                            computed_quantity=got, absolute_error=ae,
                            relative_error_if_defined=re_, cutoff_or_grid=cutoff,
                            discarded_tail_or_tolerance=(tail or tolerance),
                            tolerance=tolerance, row_kind=kind,
                            pass_fail_and_reason=("pass" if ok else "FAIL") +
                            (f"; {note}" if note else "")))
        return ok

    # ---------------------------------------------------- code construction
    z0, o0, V = st.build_codewords()
    stabs = st.build_stabilizers()
    comm = max(float(np.abs(stabs[i] @ stabs[j] - stabs[j] @ stabs[i]).max())
               for i in range(6) for j in range(6))
    stab_res = max(max(float(np.abs(S @ z0 - z0).max()),
                       float(np.abs(S @ o0 - o0).max())) for S in stabs)
    iso_res = float(np.abs(V.conj().T @ V - np.eye(2)).max())
    Gs, Pi_sum, _ = st.build_recovery_operators()
    proj_res = float(np.abs(Pi_sum - np.eye(128)).max())
    tp_res = float(np.abs(sum(G.conj().T @ G for G in Gs) - np.eye(128)).max())
    combos, classes, audit = st.enumeration_table()
    record("stabilizer commutation", "[[7,1,3]]", "explicit matrices", 0.0, comm)
    record("stabilizers fix both codewords", "[[7,1,3]]", "explicit matrices", 0.0, stab_res)
    record("isometry V^dag V = I", "[[7,1,3]]", "explicit matrices", 0.0, iso_res)
    record("sum_s Pi_s = I_128", "64 syndromes", "product formula", 0.0, proj_res)
    record("recovery+decoding trace preserving", "64 syndromes",
           "sum_s Pi_s D_s^dag V V^dag D_s Pi_s", 0.0, tp_res)
    record("residual syndrome vanishes after CSS correction", "all 4^7 strings",
           "vectorized enumeration", 0.0, float(audit["residual_nonzero_syndrome"]))
    record("logical classification: span vs commutation", "all 4^7 strings",
           "two independent classifiers", 0.0, float(audit["classification_mismatches"]))

    # -------------------------------------------- synthetic reference point
    bm = cfg["benchmark"]
    point = pr.channel_point(bm["eta_pass"], bm["nu_pass_per_polarization"],
                             bm["e0_pass"], bm["z_pass"])
    keys_id = pr.key_set("identity")
    Fe_enum, logical = pr.exact_fidelity(point, keys_id, "pauli_enumeration")
    t0 = time.time()
    Fe_dm, rho_RL = pr.exact_fidelity(point, keys_id, "density_matrix")
    dm_time = time.time() - t0

    expected = {"T": 0.166375, "nu": 0.00055575, "c0": 0.166005661666998,
                "c1": 0.000925133616550, "s": 0.166930795283549,
                "r": 0.005542018864637, "wI": 0.995843485851522,
                "b_1": 0.99699004, "b_2": 0.997, "b_3": 0.99699004,
                "q": 0.013111962893864, "C_bound": 0.996544331711809,
                "P_acc": 3.61206233911942e-6, "Y_lb": 3.59958024983916e-6,
                "Fe_exact": 0.997325267543578, "Y_exact": 3.60240103874636e-6}
    got = dict(point)
    got["b_1"], got["b_2"], got["b_3"] = point["b"]
    got["Fe_exact"] = Fe_dm
    got["Y_exact"] = point["P_acc"] * Fe_dm
    bench_rows = []
    for k, ref in expected.items():
        g = float(got[k])
        rel = abs(g - ref) / abs(ref)
        bench_rows.append(dict(quantity=k, expected=ref, computed=g,
                               absolute_error=abs(g - ref), relative_error=rel,
                               status="pass" if rel <= 1e-11 else "FAIL"))
    man.add_output(io.write_csv(os.path.join(d, "benchmark_comparison.csv"), bench_rows))
    record("synthetic reference point (all 16 tabulated quantities)", "configs/validation.yaml",
           "independently tabulated reference values", 0.0,
           max(r["relative_error"] for r in bench_rows),
           note="max relative deviation over the 16 tabulated quantities")
    record("Pauli enumeration vs 8-qubit density matrix", "benchmark point",
           "16384-string enumeration vs 256x256 recovery", Fe_enum, Fe_dm,
           cutoff="exact", note=f"density-matrix route {dm_time*1e3:.0f} ms")

    net = point["net_pauli_ixzy"]
    man.add_output(io.write_csv(
        os.path.join(d, "logical_pauli_probabilities.csv"),
        [dict(indexing="(I,X,Z,Y) with index = x + 2z", pauli=p,
              net_physical_probability=float(net[i]),
              recovered_logical_probability=float(logical[i]))
         for i, p in enumerate(["I", "X", "Z", "Y"])]))

    # ------------------------------------------- recovery verification cases
    key_cases = {
        "identity": pr.key_set("identity"),
        "commuting_rz": pr.key_set("fixed_rz", theta_A=0.7, theta_B=-1.3),
        "fixed_nonclifford_ry": pr.key_set(
            "fixed_nonclifford", axis=cfg["decoder_sweep"]["fixed_nonclifford"]["axis"],
            theta_A=cfg["decoder_sweep"]["fixed_nonclifford"]["theta_A"],
            theta_B=cfg["decoder_sweep"]["fixed_nonclifford"]["theta_B"]),
    }
    for name, keys in key_cases.items():
        record(f"private operations commute ({name})", name, "[A_j,B_j]",
               0.0, pr.check_commuting(keys))
        # no faults
        S = [lp.unitary_super(pr.location_unitary(A, B, lp.I2, lp.I2, lp.I2))
             for A, B in keys]
        Fe, _ = st.logical_entanglement_fidelity(S, Gs=pr.recovery_ops())
        record(f"no faults -> unit logical fidelity ({name})", name,
               "ordered private-operation sequence", 1.0, Fe)

    # 21 single-location non-identity Paulis, at each traversal slot
    worst = 0.0
    for name, keys in key_cases.items():
        for site in range(7):
            for pi, P in enumerate([lp.PX, lp.PY, lp.PZ]):
                for slot in range(3):
                    E = [lp.I2, lp.I2, lp.I2]
                    E[slot] = P
                    S = []
                    for j, (A, B) in enumerate(keys):
                        K = pr.location_unitary(A, B, *(E if j == site else
                                                        [lp.I2, lp.I2, lp.I2]))
                        S.append(lp.unitary_super(K))
                    Fe, _ = st.logical_entanglement_fidelity(S, Gs=pr.recovery_ops())
                    worst = max(worst, abs(1.0 - Fe))
    record("21 single-location Paulis x 3 traversal slots x 3 key cases",
           "189 cases per key set", "explicit fault insertion + CSS recovery",
           1.0, 1.0 - worst, note="worst |1 - Fe| over all inserted single-location Paulis")

    # arbitrary single-location rotations (linear-span guarantee)
    rng = np.random.default_rng(4242)
    worst_rot = 0.0
    for _ in range(40):
        site = int(rng.integers(7)); slot = int(rng.integers(3))
        ax = "xyz"[int(rng.integers(3))]
        E = [lp.I2, lp.I2, lp.I2]
        E[slot] = lp.rot(ax, float(rng.uniform(0, 2 * np.pi)))
        keys = key_cases["fixed_nonclifford_ry"]
        S = []
        for j, (A, B) in enumerate(keys):
            K = pr.location_unitary(A, B, *(E if j == site else [lp.I2] * 3))
            S.append(lp.unitary_super(K))
        Fe, _ = st.logical_entanglement_fidelity(S, Gs=pr.recovery_ops())
        worst_rot = max(worst_rot, abs(1.0 - Fe))
    record("arbitrary single-location rotations (40 random)",
           "non-Clifford keys", "linear-span correction guarantee", 1.0,
           1.0 - worst_rot)

    # same-location faults on several traversals
    worst_multi = 0.0
    for _ in range(30):
        site = int(rng.integers(7))
        E = [lp.rot("xyz"[int(rng.integers(3))], float(rng.uniform(0, 2 * np.pi)))
             for _ in range(3)]
        keys = key_cases["fixed_nonclifford_ry"]
        S = []
        for j, (A, B) in enumerate(keys):
            K = pr.location_unitary(A, B, *(E if j == site else [lp.I2] * 3))
            S.append(lp.unitary_super(K))
        Fe, _ = st.logical_entanglement_fidelity(S, Gs=pr.recovery_ops())
        worst_multi = max(worst_multi, abs(1.0 - Fe))
    record("faults on all three traversals of ONE location (30 random)",
           "non-Clifford keys", "ordered sequence + CSS recovery", 1.0,
           1.0 - worst_multi)

    # fully depolarizing diagnostic
    S_dep = [lp.pauli_channel_super([0.25, 0.25, 0.25, 0.25])] * 7
    Fe_dep, rho_dep = st.logical_entanglement_fidelity(S_dep, Gs=pr.recovery_ops())
    record("fully depolarizing physical channel", "p=(1/4,1/4,1/4,1/4)",
           "8-qubit density matrix", 0.25, Fe_dep,
           note="diagnostic only, not an operating regime")
    record("recovered state trace", "benchmark point", "8-qubit density matrix",
           1.0, float(np.real(np.trace(rho_RL))))
    record("recovered state minimum eigenvalue >= 0", "benchmark point",
           "8-qubit density matrix", 0.0,
           max(0.0, -float(np.linalg.eigvalsh(rho_RL).min())))
    record("recovered state hermiticity", "benchmark point",
           "8-qubit density matrix", 0.0,
           float(np.abs(rho_RL - rho_RL.conj().T).max()))

    # correlated-fault control (Section 14.5)
    p_corr = 0.001
    ZL = st.n_qubit_pauli(np.zeros(7, np.int8), np.ones(7, np.int8))
    Fe_corr_exact = 1.0 - p_corr
    C_iid = float(rl.correctable_weight(p_corr))
    markov = rl.markov_correlated_bound([p_corr] * 7)
    corr_rows = [dict(control="correlated Z^{(x)7} with probability p", p=p_corr,
                      exact_logical_Fe=Fe_corr_exact,
                      iid_binomial_C_7_1=C_iid,
                      iid_bound_exceeds_exact=bool(C_iid > Fe_corr_exact),
                      markov_support_bound=markov,
                      note="intentional test OUTSIDE the i.i.d. hypotheses; not a "
                           "counterexample to the theorem under its hypotheses")]
    man.add_output(io.write_csv(os.path.join(d, "correlated_fault_control.csv"), corr_rows))

    # ------------------------------------------------------- noise sweep 14.1
    ds = cfg["decoder_sweep"]
    us = np.concatenate([[0.0], np.logspace(np.log10(ds["u_min"]),
                                            np.log10(ds["u_max"]), ds["u_points"])])
    nu_ref = np.array(bm["nu_pass_per_polarization"], float)
    e0_ref = np.array(bm["e0_pass"], float)
    z_ref = np.array(bm["z_pass"], float)
    sig2_ref = lp.sigma2_from_z(z_ref)
    # Reproducible commuting random key sets.  Generation rule: for each of the
    # seven locations a COMMON axis is drawn uniformly from {x, y, z} and the two
    # rotation angles are drawn uniformly from [0, 2 pi); A_j and B_j then share
    # an axis and commute by construction.  This is NOT Haar sampling over all
    # admissible commuting key pairs, and the resulting spread is not a
    # worst-case guarantee over private keys.
    rand_keys = [pr.key_set("random", seed=ds["key_seed"] + i)
                 for i in range(ds["n_random_key_sets"])]
    record("private key sets are commuting by construction",
           f"{ds['n_random_key_sets']} random sets, seed base {ds['key_seed']}",
           "max |[A_j, B_j]| over all sets and locations", 0.0,
           float(max(pr.check_commuting(k) for k in rand_keys)),
           note="axis drawn from {x,y,z}, angles from [0,2pi); not Haar sampling")
    rand_at = set(np.linspace(0, len(us) - 1, 9).astype(int).tolist())

    # Two noise paths.  The baseline scales all components together from the
    # synthetic reference, where the differential-XPM contribution is ~1e-5 and
    # therefore barely exercises non-Clifford error propagation.  The second path
    # is XPM-DOMINATED: the base error is reduced and the differential phase
    # variance raised so that the Z-type channel carries most of the fault weight
    # and the commuting non-Clifford rotations act on a strongly anisotropic
    # channel.
    PATHS = {"baseline": (nu_ref, e0_ref, sig2_ref),
             "xpm_dominant": (0.1 * nu_ref, 0.0002 * np.ones(3),
                              np.array([0.10, 0.10, 0.10]))}

    sweep, keyrows = [], []
    for path_name, (nu_r, e0_r, sig2_r) in PATHS.items():
      for iu, u in enumerate(us):
          nu_u = u * nu_r
          e0_u = u * e0_r
          sig2_u = u * sig2_r
          z_u = lp.z_from_sigma2(sig2_u)
          pt = pr.channel_point(bm["eta_pass"], nu_u, e0_u, z_u)
          Fe_pe, _ = pr.exact_fidelity(pt, key_cases["identity"], "pauli_enumeration")
          Fe_id, _ = pr.exact_fidelity(pt, key_cases["identity"], "density_matrix")
          Fe_nc, _ = pr.exact_fidelity(pt, key_cases["fixed_nonclifford_ry"], "density_matrix")
          base = dict(noise_path=path_name, u=u, T=pt["T"], nu=pt["nu"],
                      s=pt["s"], r=pt["r"],
                      e0=e0_u[0], z_1=z_u[0], z_3=z_u[2], sigma2_1=sig2_u[0],
                      q=pt["q"], C_bound=pt["C_bound"], P_acc=pt["P_acc"],
                      Y_lb=pt["Y_lb"], Y_ub=pt["Y_ub"])
          for kname, Fe, meth in (("identity", Fe_id, "density_matrix"),
                                  ("identity", Fe_pe, "pauli_enumeration"),
                                  ("fixed_nonclifford_ry", Fe_nc, "density_matrix")):
              sweep.append(dict(base, key_case=kname, method=meth, Fe_exact=Fe,
                                Y_exact=pt["P_acc"] * Fe,
                                gap_Fe_minus_C=Fe - pt["C_bound"],
                                bound_satisfied=bool(Fe >= pt["C_bound"] - 1e-12),
                                yield_bracket_ok=bool(
                                    pt["Y_lb"] <= pt["P_acc"] * Fe + 1e-18 <= pt["Y_ub"] + 1e-18),
                                uncoded_n1_Fe=pr.uncoded_fidelity(pt)))
          sweep.append(dict(base, key_case="identity", method="pauli_vs_dm_residual",
                            Fe_exact=abs(Fe_pe - Fe_id), Y_exact=np.nan,
                            gap_Fe_minus_C=np.nan, bound_satisfied=True,
                            yield_bracket_ok=True, uncoded_n1_Fe=np.nan))
          if iu in rand_at:
              fes = []
              for ki, keys in enumerate(rand_keys):
                  Fe_r, _ = pr.exact_fidelity(pt, keys, "density_matrix")
                  fes.append(Fe_r)
                  keyrows.append(dict(noise_path=path_name, u=u, q=pt["q"],
                                      C_bound=pt["C_bound"],
                                      key_set_id=f"random_{ds['key_seed']}_{ki}",
                                      Fe_exact=Fe_r, gap_Fe_minus_C=Fe_r - pt["C_bound"],
                                      bound_satisfied=bool(Fe_r >= pt["C_bound"] - 1e-12)))
              keyrows.append(dict(noise_path=path_name, u=u, q=pt["q"],
                                  C_bound=pt["C_bound"],
                                  key_set_id="random_range", Fe_exact=np.nan,
                                  gap_Fe_minus_C=np.nan, bound_satisfied=True,
                                  Fe_min=min(fes), Fe_max=max(fes)))

    man.add_output(io.write_csv(os.path.join(d, "decoder_validation.csv"), sweep))
    man.add_output(io.write_csv(
        os.path.join(d, "key_sensitivity.csv"), keyrows,
        ["noise_path", "u", "q", "C_bound", "key_set_id", "Fe_exact",
         "gap_Fe_minus_C", "bound_satisfied", "Fe_min", "Fe_max"]))

    viol = [r for r in sweep if not r["bound_satisfied"]]
    ybad = [r for r in sweep if not r["yield_bracket_ok"]]
    pvd = max(r["Fe_exact"] for r in sweep if r["method"] == "pauli_vs_dm_residual")
    record("Fe_exact >= C_bound over the whole sweep",
           f"{len(sweep)} sweep rows", "exact recovery vs binomial certificate",
           0.0, float(len(viol)), note="count of violations")
    record("Y_lb <= Y_exact <= Y_ub over the whole sweep",
           f"{len(sweep)} sweep rows", "exact recovery vs yield bracket",
           0.0, float(len(ybad)), note="count of violations")
    record("Pauli enumeration vs density matrix across the sweep",
           f"{len(us)} noise points", "two independent exact methods", 0.0, pvd,
           note="max |Fe_enum - Fe_dm|")
    record("random key sets satisfy the certificate",
           f"{len(keyrows)} rows", "8-qubit density matrix", 0.0,
           float(sum(1 for r in keyrows if not r["bound_satisfied"])),
           note="count of violations")

    man.add_output(io.write_csv(os.path.join(d, "validation_table_s2_decoder.csv"), rows_s2))
    man.save(os.path.join(d, "run_manifest_decoder.json"))

    checks = [r for r in rows_s2 if r["row_kind"] == "assert"]
    diags = [r for r in rows_s2 if r["row_kind"] != "assert"]
    nfail = sum(1 for r in checks if r["pass_fail_and_reason"].startswith("FAIL"))
    print(f"\nStage A+C complete: {len(checks)} asserted checks, {nfail} failures, "
          f"{len(diags)} report-only diagnostics, {len(sweep)} sweep rows over "
          f"{len(us)} noise points x {len(PATHS)} paths")
    for r in rows_s2:
        print(f"  [{r['pass_fail_and_reason'][:4]:4s}] {r['validation_case']:<62s} "
              f"err={r['absolute_error']:.3e}")
    return nfail


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
