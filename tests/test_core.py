"""Fast regression tests for the scientific core (the regression tests).

Run:  python -m pytest tests -q      (from the project root)
The heavier independent validations live in scripts/validate_optics.py and
scripts/validate_decoder.py, which write the achieved residuals to CSV.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import attenuation as at
import deployment as dpl
import fwm as fw
import local_protocol as lp
import modes as md
import optical_channel as oc
import optical_dilation as od
import protocol as pr
import raman as rm
import reliability as rl
import steane as st
import xpm as xp
from units import (dbm_to_w, w_to_dbm, nm_to_hz, hz_to_nm, db_per_km_to_per_m,
                   per_m_to_db_per_km, H_PLANCK)

BENCH = dict(eta=[0.55] * 3, nu=[3e-4] * 3, e0=0.002, z=[1e-5, 0.0, 1e-5])


# ----------------------------------------------------------------------- units
def test_unit_round_trips():
    assert w_to_dbm(dbm_to_w(-5.0)) == pytest.approx(-5.0, abs=1e-12)
    assert hz_to_nm(nm_to_hz(1550.0)) == pytest.approx(1550.0, rel=1e-15)
    assert per_m_to_db_per_km(db_per_km_to_per_m(0.19)) == pytest.approx(0.19, rel=1e-15)
    assert np.exp(-db_per_km_to_per_m(0.2) * 1e4) == pytest.approx(10 ** -0.2, rel=1e-14)


# ----------------------------------------------------------------------- modes
def test_mode_normalization_and_polarization_factor():
    m = md.GaussianMode(nm_to_hz(1550.0), 10e9)
    assert m.norm_check(48) == pytest.approx(1.0, abs=1e-13)
    f0 = m.f0
    nu = md.occupancy_from_spectrum(m, lambda f: np.full_like(f, 1e-20))
    assert nu == pytest.approx(0.5 * 1e-20 / (H_PLANCK * f0), rel=1e-5)
    nu_pp = md.occupancy_from_spectrum(m, lambda f: np.full_like(f, 1e-20),
                                       per_polarization=True)
    assert nu_pp / nu == pytest.approx(2.0, rel=1e-14)


def test_equivalent_noise_bandwidth_is_not_the_fwhm():
    f0 = nm_to_hz(1550.0)
    filt = md.SuperGaussianFilter(f0, 25e9, 4, 110.0,
                                  (nm_to_hz(1600.0), nm_to_hz(1500.0)))
    assert filt(f0) == pytest.approx(1.0, abs=1e-10)
    assert filt(f0 + 12.5e9) == pytest.approx(0.5, abs=1e-10)
    beq = filt.equivalent_noise_bandwidth()
    # B_eq is a finite integral over the modeled domain, not the FWHM.  For this
    # 4th-order super-Gaussian the two differ by only ~1.4%, so substituting the
    # FWHM (as the legacy code did) is a small error for THIS filter shape; the
    # point is that they are different quantities and B_eq needs a finite domain.
    assert beq != pytest.approx(25e9, rel=1e-4)
    assert beq == pytest.approx(24.647e9, rel=1e-3)
    with pytest.raises(ValueError):
        md.SuperGaussianFilter(f0, 25e9, 4, 110.0).equivalent_noise_bandwidth()


# ----------------------------------------------------------------------- Raman
def test_raman_closed_forms_match_longitudinal_integration():
    a1, a2, L = 4.5e-5, 4.4e-5, 1e4
    num = rm.longitudinal_check(a1, a2, L, True)
    ana = float(rm._long_factor(np.array(a1 - a2), L)) * float(np.exp(-a2 * L))
    assert ana == pytest.approx(num, rel=1e-9)
    num = rm.longitudinal_check(a1, a2, L, False)
    ana = float(rm._long_factor(np.array(a1 + a2), L))
    assert ana == pytest.approx(num, rel=1e-9)
    assert rm.longitudinal_check(a1, a1, L, True) == pytest.approx(
        L * np.exp(-a1 * L), rel=1e-9)


def test_raman_is_calibrated_at_the_reported_point_not_the_peak():
    """The reference is of order 1e-23 in SI units, so the comparison MUST be
    relative with the absolute tolerance switched off: pytest.approx's default
    absolute tolerance of 1e-12 would accept any coefficient of this size."""
    from units import rho_km_nm_to_per_m_per_hz
    ref = rho_km_nm_to_per_m_per_hz(rm.QI_RHO_KM_NM, rm.QI_OBS_NM)
    assert ref < 1e-20                               # dimensionful and tiny
    assert float(rm.calibrated_rho(rm.QI_DELTA_F_HZ)) == pytest.approx(
        ref, rel=1e-13, abs=0.0)
    assert rm.QI_DELTA_F_HZ < 0                      # anti-Stokes, ~-0.6 THz
    assert float(rm.calibrated_rho(13.2e12)) > 2 * ref


def test_raman_calibration_check_rejects_a_wrong_normalization():
    """Fault injection: a coefficient inflated by 1e6 must FAIL the calibration
    comparison.  Under the earlier absolute-tolerance comparison it passed."""
    from units import rho_km_nm_to_per_m_per_hz
    ref = rho_km_nm_to_per_m_per_hz(rm.QI_RHO_KM_NM, rm.QI_OBS_NM)
    wrong = float(rm.calibrated_rho(rm.QI_DELTA_F_HZ)) * 1e6
    assert wrong != pytest.approx(ref, rel=1e-13, abs=0.0)      # correct check
    assert wrong == pytest.approx(ref, abs=1e-12)               # earlier check
    assert wrong / ref == pytest.approx(1e6, rel=1e-9)


def test_fwm_degenerate_benchmark_and_factor_of_nine_rejection():
    """Independent low-conversion benchmark: P = gamma^2 P_p^2 P_s L^2, computed
    here without the production prefactor.  The earlier convention (d = 3, 6
    with the conventional gamma) gives nine times that power and is rejected."""
    class _Lossless:
        valid_nm = (1000.0, 2000.0)

        def a_per_m(self, lam_nm, allow_extrapolation=False):
            return np.zeros_like(np.asarray(lam_nm, dtype=float))

    flat = at.Dispersion(beta2_s2_m=0.0, beta3_s3_m=0.0, n_group=1.468)
    g, L, Pp, Ps = 1.3e-3, 1.0e4, 1.0e-3, 2.0e-4
    dep = dpl.Deployment(freqs_hz=np.array([193.0e12, 193.1e12]),
                         launch_w=np.array([Pp, Ps]))
    p = fw.enumerate_products(dep, flat, _Lossless(), L, g)
    i = int(np.argmin(np.abs(p.freq_hz - (2 * 193.0e12 - 193.1e12))))
    expect = g ** 2 * Pp ** 2 * Ps * L ** 2
    assert float(p.power_incoherent_w[i]) == pytest.approx(expect, rel=1e-8, abs=0.0)
    bad = fw.enumerate_products(dep, flat, _Lossless(), L, 3.0 * g,
                                convention="gamma_over_three")
    j = int(np.argmin(np.abs(bad.freq_hz - (2 * 193.0e12 - 193.1e12))))
    assert float(bad.power_incoherent_w[j]) / expect == pytest.approx(9.0, rel=1e-8)
    assert float(bad.power_incoherent_w[j]) != pytest.approx(expect, rel=1e-8, abs=0.0)


def test_fwm_nondegenerate_benchmark():
    class _Lossless:
        valid_nm = (1000.0, 2000.0)

        def a_per_m(self, lam_nm, allow_extrapolation=False):
            return np.zeros_like(np.asarray(lam_nm, dtype=float))

    flat = at.Dispersion(beta2_s2_m=0.0, beta3_s3_m=0.0, n_group=1.468)
    g, L = 1.3e-3, 1.0e4
    P = np.array([1.0e-3, 7.0e-4, 3.0e-4])
    dep = dpl.Deployment(freqs_hz=np.array([193.0e12, 193.1e12, 193.3e12]),
                         launch_w=P)
    p = fw.enumerate_products(dep, flat, _Lossless(), L, g)
    f_out = 193.0e12 + 193.1e12 - 193.3e12
    i = int(np.argmin(np.abs(p.freq_hz - f_out)))
    assert float(p.power_incoherent_w[i]) == pytest.approx(
        4.0 * g ** 2 * P[0] * P[1] * P[2] * L ** 2, rel=1e-8, abs=0.0)


def test_receiver_mode_conventions_and_profiled_lines():
    mode = md.GaussianMode(193.4e12, 10e9)
    filt = md.SuperGaussianFilter(193.4e12, 25e9, 4, 110.0,
                                  (nm_to_hz(1600.0), nm_to_hz(1500.0)))
    ov = md.mode_overlap_transmission(mode, lambda f: np.sqrt(filt(f)))
    pa = md.power_average_transmission(mode, filt)
    assert ov < pa                      # the overlap is the smaller quantity
    assert ov == pytest.approx(0.990210, rel=1e-5)
    assert pa == pytest.approx(0.991466, rel=1e-5)
    assert md.mode_overlap_transmission(
        mode, lambda f: np.sqrt(0.7) * np.ones_like(f)) == pytest.approx(0.7, rel=1e-13)
    d0 = md.occupancy_from_profiled_lines(mode, [1e-12], [193.4e12], 0.0)
    assert d0 == pytest.approx(md.occupancy_from_line([1e-12], [193.4e12], mode),
                               rel=1e-14, abs=0.0)
    wide = md.occupancy_from_profiled_lines(mode, [1e-12], [193.4e12], 25e9)
    assert wide < d0


# ------------------------------------------------------------------------- FWM
def test_fwm_degeneracy_and_lossy_amplitude():
    dep = dpl.Deployment(freqs_hz=np.array([193.0e12, 193.1e12, 193.2e12]),
                         launch_w=np.full(3, dbm_to_w(-5.0)))
    disp = at.Dispersion(beta2_s2_m=-21.7e-27, beta3_s3_m=0.12e-39)
    atten = at.AttenuationModel()
    p = fw.enumerate_products(dep, disp, atten, 1e4, 1.3e-3, quad_check=9)
    expect = sum(1 for i in range(3) for j in range(i, 3) for k in range(3)
                 if k != i and k != j)
    assert p.n_triplets == expect
    assert p.audit["closed_vs_quadrature_max_rel"] < 1e-4


# ------------------------------------------------------------------------- XPM
def test_xpm_zero_cases_and_limits():
    dep = dpl.Deployment(freqs_hz=np.array([193.0e12, 193.5e12]),
                         launch_w=np.full(2, dbm_to_w(-5.0)))
    a = np.full(2, 4.4e-5)
    assert xp.sigma2_closed(dep, a, np.full(2, 1e-13), 1e4, 1.3e-3, 0.0, 1.0, 25e9) == 0.0
    assert xp.sigma2_closed(dep, a, np.full(2, 1e-13), 1e4, 1.3e-3, 1.0, 0.0, 25e9) == 0.0
    # zero walk-off reduces to L_eff^2
    L, aa = 1e4, 4.4e-5
    got = xp.sigma2_closed(dep, a, np.zeros(2), L, 1.3e-3, 1.0, 1.0, 25e9)
    leff2 = ((1 - np.exp(-aa * L)) / aa) ** 2
    want = 4 * (1.3e-3) ** 2 * np.sum(dep.launch_w ** 2) * leff2
    assert got == pytest.approx(want, rel=1e-10)


def test_xpm_closed_form_matches_both_reference_routes():
    dep = dpl.Deployment(freqs_hz=np.array([193.0e12, 193.4e12, 193.8e12]),
                         launch_w=np.full(3, dbm_to_w(-5.0)))
    disp = at.Dispersion(beta2_s2_m=-21.7e-27, beta3_s3_m=0.12e-39)
    a = np.full(3, 4.4e-5)
    d = disp.walkoff_co(dep.freqs_hz, nm_to_hz(1550.0))
    args = (1e4, 1.3e-3, 1.0, 1.0, 25e9)
    cl = xp.sigma2_closed(dep, a, d, *args)
    sp, resolved = xp.sigma2_spectral(dep, a, d, *args, n_F=200001)
    cv = xp.sigma2_covariance(dep, a, d, *args, n_z=3001)
    assert resolved
    assert cl == pytest.approx(sp, rel=1e-6)
    assert cl == pytest.approx(cv, rel=1e-4)


# ------------------------------------------------------- optical reduction
def test_optical_reduction_limits():
    assert float(oc.accepted_optical(0.37, 0.0).s) == pytest.approx(0.37, rel=1e-15)
    assert float(oc.accepted_optical(0.37, 0.0).r) == pytest.approx(0.0, abs=1e-15)
    assert float(oc.accepted_optical(1.0, 0.0).s) == pytest.approx(1.0, rel=1e-15)
    assert float(oc.accepted_optical(0.0, 1e-3).r) == pytest.approx(1.0, rel=1e-15)
    assert float(oc.accepted_optical(0.05, 0.05).s) > 0.05      # s can exceed T
    assert not bool(oc.accepted_optical(0.0, 0.0).defined)


def test_background_composition_weights():
    _, nu, _ = oc.compose_passes([0.5] * 3, [1.0, 0.0, 0.0])
    assert nu == pytest.approx(0.25, rel=1e-15)
    _, nu, _ = oc.compose_passes([0.5] * 3, [0.0, 0.0, 1.0])
    assert nu == pytest.approx(1.0, rel=1e-15)


@pytest.mark.parametrize("T,nu", [(0.05, 1e-1), (0.2, 1e-3), (0.6, 0.0), (0.9, 1e-2)])
def test_dilation_reproduces_the_closed_form(T, nu):
    J, Ech, info = od.accepted_choi_numeric(T, nu)
    res = oc.accepted_optical(T, nu)
    Jf = oc.choi_from_weights(float(res.c0), float(res.c1))
    assert float(np.linalg.svd(J - Jf, compute_uv=False).sum()) < 1e-12
    accs = []
    for rho in (np.array([[1, 0], [0, 0]], complex),
                0.5 * np.array([[1, -1j], [1j, 1]], complex)):
        accs.append(float(np.real(np.trace(od.accepted_map_numeric(T, nu, rho, Ech=Ech)))))
    assert accs[0] == pytest.approx(accs[1], abs=1e-14)


def test_three_traversals_match_one_effective_channel():
    eta, nup = [0.8, 0.7, 0.6], [1e-4, 2e-4, 3e-4]
    T, nu, _ = oc.compose_passes(eta, nup)
    J3, _ = od.three_traversal_accepted(eta, nup, dim=7)
    res = oc.accepted_optical(T, nu)
    Jf = oc.choi_from_weights(float(res.c0), float(res.c1))
    assert float(np.linalg.svd(J3 - Jf, compute_uv=False).sum()) < 1e-12


# ------------------------------------------------------------------ the code
def test_steane_construction_and_recovery():
    z0, o0, V = st.build_codewords()
    stabs = st.build_stabilizers()
    assert np.abs(V.conj().T @ V - np.eye(2)).max() < 1e-14
    for A in stabs:
        for B in stabs:
            assert np.abs(A @ B - B @ A).max() == 0.0
        assert np.abs(A @ z0 - z0).max() == 0.0
        assert np.abs(A @ o0 - o0).max() == 0.0
    Gs, Pi_sum, _ = st.build_recovery_operators()
    assert np.abs(Pi_sum - np.eye(128)).max() < 1e-13
    assert np.abs(sum(G.conj().T @ G for G in Gs) - np.eye(128)).max() < 1e-13
    _, _, audit = st.enumeration_table()
    assert audit["residual_nonzero_syndrome"] == 0
    assert audit["classification_mismatches"] == 0
    assert audit["n_strings"] == 4 ** 7


def test_all_21_single_location_paulis_are_corrected():
    keys = pr.key_set("fixed_nonclifford", axis="y", theta_A=0.63, theta_B=1.05)
    for site in (0, 3, 6):
        for P in (lp.PX, lp.PY, lp.PZ):
            for slot in range(3):
                E = [lp.I2] * 3
                E[slot] = P
                S = [lp.unitary_super(pr.location_unitary(
                    A, B, *(E if j == site else [lp.I2] * 3)))
                    for j, (A, B) in enumerate(keys)]
                Fe, _ = st.logical_entanglement_fidelity(S, Gs=pr.recovery_ops())
                assert Fe == pytest.approx(1.0, abs=1e-12)


def test_fully_depolarizing_gives_one_quarter():
    S = [lp.pauli_channel_super([0.25] * 4)] * 7
    Fe, _ = st.logical_entanglement_fidelity(S, Gs=pr.recovery_ops())
    assert Fe == pytest.approx(0.25, abs=1e-12)


# ---------------------------------------------------------------- benchmark
def test_section_12_benchmark():
    pt = pr.channel_point(BENCH["eta"], BENCH["nu"], BENCH["e0"], BENCH["z"])
    exp = {"T": 0.166375, "nu": 0.00055575, "c0": 0.166005661666998,
           "c1": 0.000925133616550, "s": 0.166930795283549,
           "r": 0.005542018864637, "wI": 0.995843485851522,
           "q": 0.013111962893864, "C_bound": 0.996544331711809,
           "P_acc": 3.61206233911942e-6, "Y_lb": 3.59958024983916e-6}
    for k, v in exp.items():
        assert float(pt[k]) == pytest.approx(v, rel=1e-11), k
    assert np.allclose(pt["b"], [0.99699004, 0.997, 0.99699004], rtol=1e-9)
    fe_enum, _ = pr.exact_fidelity(pt, pr.key_set("identity"), "pauli_enumeration")
    fe_dm, _ = pr.exact_fidelity(pt, pr.key_set("identity"), "density_matrix")
    assert fe_enum == pytest.approx(0.997325267543578, rel=1e-11)
    assert fe_dm == pytest.approx(fe_enum, abs=1e-12)
    assert pt["P_acc"] * fe_enum == pytest.approx(3.60240103874636e-6, rel=1e-11)
    assert fe_enum >= pt["C_bound"]
    assert pt["Y_lb"] <= pt["P_acc"] * fe_enum <= pt["Y_ub"]


def test_uncoded_comparison_is_not_one_minus_q():
    pt = pr.channel_point(BENCH["eta"], BENCH["nu"], BENCH["e0"], BENCH["z"])
    assert pr.uncoded_fidelity(pt) != pytest.approx(1 - pt["q"], rel=1e-6)


# ------------------------------------------------------------------- bounds
def test_binomial_stability_and_reference():
    q = np.array([0.0, 1e-12, 1e-6, 1e-3, 0.1, 0.5, 1.0])
    assert np.abs(rl.correctable_weight_steane(q) -
                  rl.correctable_weight_binom(q)).max() < 1e-14
    assert rl.uncorrectable_weight_steane(1e-12) == pytest.approx(21e-24, rel=1e-6)
    assert 1 - rl.correctable_weight_steane(1e-12) <= 0     # naive form cancels


def test_correlated_control_breaks_the_iid_expression():
    p = 0.001
    assert float(rl.correctable_weight(p)) > 1 - p          # iid exceeds the exact value
    assert rl.markov_correlated_bound([p] * 7) == pytest.approx(1 - 7 * p / 2, rel=1e-12)


# --------------------------------------------------------- data / provenance
def test_attenuation_rejects_unsupported_extrapolation():
    a = at.AttenuationModel()
    with pytest.raises(ValueError):
        a.db_per_km(1200.0)


def test_probability_domains_are_asserted():
    with pytest.raises(ValueError):
        oc.accepted_optical(1.5, 0.0)
    with pytest.raises(ValueError):
        oc.accepted_optical(0.5, -1e-3)
    with pytest.raises(ValueError):
        lp.traversal_weights(1.0, 0.0)                      # a0 = 0.5 > 1/3
    with pytest.raises(ValueError):
        lp.z_from_sigma2(np.array(-1.0))


def test_hollenbeck_cantrell_model_is_the_published_lineshape():
    """The intermediate-broadening fit peaks at the silica Raman maximum
    (~440 cm^-1), is nonnegative, is normalized exactly at the Qi et al. point,
    and obeys detailed balance between the Stokes and anti-Stokes sides."""
    from units import C_LIGHT, H_PLANCK, K_BOLTZ, rho_km_nm_to_per_m_per_hz
    df, g = rm._hc_gain_table()
    assert df[np.argmax(g)] / (100.0 * C_LIGHT) == pytest.approx(440.0, rel=5e-3)
    assert np.all(g >= 0.0)
    ref = rho_km_nm_to_per_m_per_hz(rm.QI_RHO_KM_NM, rm.QI_OBS_NM)
    assert float(rm.calibrated_rho(rm.QI_DELTA_F_HZ)) == pytest.approx(
        ref, rel=1e-13, abs=0.0)
    x = 0.9e12
    ratio = float(rm.calibrated_rho(-x) / rm.calibrated_rho(x))
    assert ratio == pytest.approx(np.exp(-H_PLANCK * x / (K_BOLTZ * 300.0)), rel=1e-12)

