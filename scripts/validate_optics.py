#!/usr/bin/env python
"""Stage B: independent optical validation and the impairment-model unit tests
(Methods: optical impairment models)."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import attenuation as at
import config as cf
import deployment as dpl
import fwm as fw
import leakage as lk
import legacy_compare as lg
import local_protocol as lp
import modes as md
import optical_channel as oc
import optical_dilation as od
import raman as rm
import runio as io
import xpm as xp
from units import (C_LIGHT, H_PLANCK, K_BOLTZ, dbm_to_w, w_to_dbm, nm_to_hz,
                   hz_to_nm, db_per_km_to_per_m, per_m_to_db_per_km,
                   rho_km_nm_to_per_m_per_hz)

RUN_ID = os.environ.get("QCFC_RUN_ID", "validation")
TOL_ABS, TOL_COND = 1e-9, 1e-7
rows = []


def rec(case, params, method, ref, got, cutoff="", tail="", tol=None,
        rel=None, abs_tol=None, note="", kind="assert"):
    """Record an ASSERTED check.

    A tolerance must be stated as a relative tolerance (`rel`, or the legacy
    `tol` when the reference is nonzero) or as an explicit absolute tolerance
    (`abs_tol`, or `tol` when the reference is exactly zero).  A dimensionful
    absolute tolerance is never applied to a nonzero reference: on a quantity of
    order 1e-23 that would certify an arbitrarily large relative error, which is
    exactly how the earlier Raman calibration check passed a coefficient
    inflated by 1e6 (Methods: numerical tolerances).
    """
    ae = abs(got - ref)
    re_ = ae / abs(ref) if ref != 0 else np.nan
    if rel is None and abs_tol is None:
        if tol is None:
            raise ValueError("state a tolerance: rel=, abs_tol= or tol=")
        if ref == 0:
            abs_tol = tol
        else:
            rel = tol
    if rel is not None:
        ok = np.isfinite(re_) and re_ <= rel
        tolerance = f"rel<={rel:g}"
    else:
        ok = ae <= abs_tol
        tolerance = f"abs<={abs_tol:g}"
    rows.append(dict(validation_case=case, parameters_or_config_id=params,
                     independent_method=method, reference_quantity=ref,
                     computed_quantity=got, absolute_error=ae,
                     relative_error_if_defined=re_, cutoff_or_grid=cutoff,
                     discarded_tail_or_tolerance=(tail or tolerance),
                     tolerance=tolerance, row_kind=kind,
                     pass_fail_and_reason=("pass" if ok else "FAIL") +
                     (f"; {note}" if note else "")))
    return ok


def report(case, params, method, ref, got, note, cutoff=""):
    """Record a REPORT-ONLY diagnostic: no tolerance, never counted as a check."""
    rows.append(dict(validation_case=case, parameters_or_config_id=params,
                     independent_method=method, reference_quantity=ref,
                     computed_quantity=got,
                     absolute_error=(abs(got - ref) if isinstance(ref, float)
                                     and isinstance(got, float) else np.nan),
                     relative_error_if_defined=np.nan, cutoff_or_grid=cutoff,
                     discarded_tail_or_tolerance="", tolerance="none (diagnostic)",
                     row_kind="diagnostic",
                     pass_fail_and_reason="reported; " + note))


def main():
    cfg = cf.load("validation.yaml")
    base = cf.load("illustrative_baseline.yaml")
    d = io.run_dir(RUN_ID)
    man = io.Manifest(RUN_ID, "B optical validation",
                      {"validation": cfg, "baseline": base})
    man.tolerance("optical_absolute", TOL_ABS)
    man.tolerance("optical_conditional", TOL_COND)

    # ------------------------------------------------------------------ units
    rec("dBm/W round trip", "-5 dBm", "analytic inverse", -5.0,
        float(w_to_dbm(dbm_to_w(-5.0))), rel=1e-13)
    rec("frequency/wavelength round trip", "1550 nm", "analytic inverse", 1550.0,
        float(hz_to_nm(nm_to_hz(1550.0))), rel=1e-13)
    rec("dB/km to m^-1 round trip", "0.19 dB/km", "analytic inverse", 0.19,
        float(per_m_to_db_per_km(db_per_km_to_per_m(0.19))), rel=1e-13)
    rec("0.2 dB/km over 10 km equals 2 dB loss", "exp(-aL)", "closed form",
        10.0 ** (-0.2), float(np.exp(-db_per_km_to_per_m(0.2) * 1e4)), tol=1e-12)
    lam = 1554.94e-9
    rec("spectral Jacobian |dlambda_nm/df|", "1554.94 nm", "1e9 lambda^2/c",
        1e9 * lam ** 2 / C_LIGHT,
        float(rho_km_nm_to_per_m_per_hz(1.0, 1554.94) * 1e3), tol=1e-12)

    # ------------------------------------------------------------------ modes
    f0 = nm_to_hz(1550.0)
    mode = md.GaussianMode(f0, 10e9)
    rec("accepted-mode normalization (order 48)", "10 GHz FWHM",
        "Gauss-Hermite quadrature of |psi|^2", 1.0, mode.norm_check(48), tol=1e-12)
    rec("accepted-mode normalization (order 96)", "10 GHz FWHM",
        "Gauss-Hermite quadrature of |psi|^2", 1.0, mode.norm_check(96), tol=1e-12)
    rec("transform-limited time-bandwidth product", "10 GHz FWHM",
        "0.4413/FWHM", 0.4413 / 10e9, mode.time_fwhm_s, tol=1e-12)
    S_flat = 1e-20
    nu_flat = md.occupancy_from_spectrum(mode, lambda f: np.full_like(f, S_flat))
    rec("per-polarization factor on a flat spectrum", "S = 1e-20 W/Hz summed over H,V",
        "nu = S/(2 h f)", 0.5 * S_flat / (H_PLANCK * f0), nu_flat, tol=1e-6)
    nu_pp = md.occupancy_from_spectrum(mode, lambda f: np.full_like(f, S_flat),
                                       per_polarization=True)
    rec("per-polarization input omits the factor 1/2", "same spectrum",
        "ratio of the two conventions", 2.0, nu_pp / nu_flat, tol=1e-12)
    filt = md.SuperGaussianFilter(f0, 25e9, 4, 110.0,
                                  (nm_to_hz(1600.0), nm_to_hz(1500.0)))
    rec("filter peak response", "super-Gaussian order 4", "T_q(f0)", 1.0,
        float(filt(f0)), tol=1e-10)
    rec("filter half power at FWHM/2", "25 GHz FWHM", "T_q(f0 + 12.5 GHz)", 0.5,
        float(filt(f0 + 12.5e9)), tol=1e-10)
    Beq = filt.equivalent_noise_bandwidth()
    report("equivalent noise bandwidth is not the FWHM",
           "25 GHz FWHM, 110 dB floor, 1500-1600 nm", "B_eq = int T_q(f) df",
           25e9, Beq, f"B_eq = {Beq/1e9:.4f} GHz vs FWHM 25 GHz; the legacy code "
                      f"multiplied the Raman coefficient by the FWHM")
    mu = md.gate_photon_mean(lambda f: np.full_like(f, S_flat), 100e-12,
                             f0 - 12.5e9, f0 + 12.5e9)
    report("gate photon mean is not the mode occupancy",
           "flat 1e-20 W/Hz, 25 GHz, 100 ps gate", "mu = tau_g int S/(hf) df",
           nu_flat, mu, f"mu_gate/nu = {mu/nu_flat:.4f}; a gate mean counts "
                        f"B*tau_g temporal modes and must not replace nu")

    # ------------------------------------------------------------------ Raman
    a1, a2, L = 4.5e-5, 4.4e-5, 1e4
    for co in (True, False):
        num = rm.longitudinal_check(a1, a2, L, co)
        ana = float(rm._long_factor(np.array(a1 - a2 if co else a1 + a2), L)) * \
            (float(np.exp(-a2 * L)) if co else 1.0)
        rec(f"Raman longitudinal factor, {'co' if co else 'counter'}-propagating",
            "a_ci=4.5e-5 /m, a_obs=4.4e-5 /m, L=10 km",
            "direct longitudinal integration", num, ana, cutoff="n=200001", tol=1e-9)
    rec("Raman equal-loss limit -> L exp(-aL)", "a_ci = a_obs",
        "direct longitudinal integration", rm.longitudinal_check(a1, a1, L, True),
        float(L * np.exp(-a1 * L)), tol=1e-12)
    dep0 = cf.build_deployment(base)
    dep_zero = dpl.Deployment(freqs_hz=dep0.freqs_hz,
                              launch_w=np.zeros_like(dep0.launch_w))
    rec("zero classical launch power gives zero Raman", "P_i0 = 0", "span_spectrum",
        0.0, float(rm.span_spectrum(np.array([f0]), dep_zero, np.array([4.4e-5]),
                                    np.full(35, 4.4e-5), L, True)[0]), tol=0.0)
    rho_cal = float(rm.calibrated_rho(rm.QI_DELTA_F_HZ))
    rec("Raman normalization reproduces the reported measurement point",
        f"delta_f = {rm.QI_DELTA_F_HZ/1e12:.4f} THz (anti-Stokes), obs 1554.94 nm",
        "rho_meas converted with the spectral Jacobian",
        float(rho_km_nm_to_per_m_per_hz(rm.QI_RHO_KM_NM, rm.QI_OBS_NM)), rho_cal,
        tol=1e-12, note="normalized at the ACTUAL separation, not the 13.2 THz peak")
    # published lineshape: the intermediate-broadening fit must reproduce the
    # well-known silica gain maximum near 440 cm^-1 (13.2 THz)
    df_tab, g_tab = rm._hc_gain_table()
    peak_cm = float(df_tab[np.argmax(g_tab)] / (100.0 * C_LIGHT))
    rec("Hollenbeck-Cantrell gain maximum lies at the silica Raman peak",
        "13-mode intermediate-broadening model, JOSA B 19, 2886 (2002)",
        "known silica gain maximum near 440 cm^-1", 440.0, peak_cm, rel=5e-3,
        note="tolerance covers the 0.002 THz tabulation step")
    # independent near-carrier measurement (report only: two published
    # sources, not a numerical check)
    ratio_level = float(rm.dasilva_rho(abs(rm.QI_DELTA_F_HZ))
                        / rm.calibrated_rho(rm.QI_DELTA_F_HZ))
    d_lo, d_hi = 0.1e12, rm.DASILVA_MAX_OFFSET_HZ
    sym = lambda x: 0.5 * (rm.calibrated_rho(x) + rm.calibrated_rho(-x))
    shape_lo = float((sym(d_lo) / sym(0.6e12)) / (rm.dasilva_rho(d_lo) / rm.dasilva_rho(0.6e12)))
    shape_hi = float((sym(d_hi) / sym(0.6e12)) / (rm.dasilva_rho(d_hi) / rm.dasilva_rho(0.6e12)))
    report("Raman model versus the near-carrier measurement of da Silva et al. (2014)",
           "SMF-28, 0.1-1.4 THz carrier offsets, average of the four measured slopes",
           "published model (Qi-normalized) vs measured linear slopes",
           1.0, ratio_level,
           f"absolute level: measurement/model = {ratio_level:.3f} at 0.6 THz; "
           f"shape (both normalized at 0.6 THz): model/measurement = "
           f"{shape_lo:.2f} at 0.1 THz and {shape_hi:.2f} at 1.4 THz.  The two "
           f"published sources disagree; the measured level is carried as a "
           f"sensitivity scenario")
    rec("Stokes/anti-Stokes detailed balance at 300 K", "|delta_f| = 0.6 THz",
        "exp(h df / kT)",
        float(np.exp(H_PLANCK * 0.6e12 / (K_BOLTZ * 300.0))),
        float(rm.thermal_factor(0.6e12) / rm.thermal_factor(-0.6e12)), tol=1e-10)
    for thz in (-0.6, 1.0, 2.0, 4.0, 13.2):
        report("Raman recalibration versus the legacy normalization",
               f"delta_f = {thz} THz", "rho_new / rho_legacy",
               float(lg.legacy_raman_rho(thz * 1e12)),
               float(rm.calibrated_rho(thz * 1e12)),
               f"ratio = {float(lg.raman_recalibration_factor(thz*1e12)):.2f}x; the "
               f"legacy code assigned RAMAN_RHO_LIT to the 13.2 THz peak")
    r_alt = float(rm.calibrated_rho(4e12, model="damped_oscillator_v1") /
                  rm.calibrated_rho(4e12, model="multi_gaussian_v1"))
    report("Raman lineshape model sensitivity", "delta_f = 4 THz",
           "two stand-in shapes, both normalized at the measurement point",
           1.0, r_alt, f"alternative/baseline = {r_alt:.3f}; neither shape is a fit "
                       f"to a published measured spectrum")

    # ------------------------------------------------- receiver mode convention
    mode_r = md.GaussianMode(193.4e12, base["link"]["mode_fwhm_ghz"] * 1e9)
    filt_r = md.SuperGaussianFilter(
        193.4e12, base["link"]["filter_fwhm_ghz"] * 1e9,
        int(base["link"]["filter_order"]), base["link"]["filter_isolation_db"],
        (nm_to_hz(base["link"]["modeled_domain_nm"][1]),
         nm_to_hz(base["link"]["modeled_domain_nm"][0])))
    ov = md.mode_overlap_transmission(mode_r, lambda f: np.sqrt(filt_r(f)))
    pa = md.power_average_transmission(mode_r, filt_r)
    rec("flat transfer: modal overlap equals the power average",
        "t(f) = sqrt(0.7) constant", "analytic identity", 0.7,
        md.mode_overlap_transmission(mode_r, lambda f: np.sqrt(0.7) *
                                     np.ones_like(f)), rel=1e-13)
    report("receiver convention: modal overlap versus power average",
           f"{base['link']['mode_fwhm_ghz']:.0f} GHz mode, "
           f"{base['link']['filter_fwhm_ghz']:.0f} GHz order-"
           f"{int(base['link']['filter_order'])} filter",
           "|int |psi|^2 sqrt(T) df|^2 versus int |psi|^2 T df", pa, ov,
           f"overlap {ov:.6f} vs power average {pa:.6f}: they differ by "
           f"{100*(pa-ov)/ov:.3f}% for this filter.  The main results use the "
           f"fixed-mode overlap for BOTH the signal and the noise projection")
    # profiled-line occupancy: delta limit and the Gaussian convolution identity
    P_l, f_l = 1e-12, 193.4e12
    nu_delta = md.occupancy_from_profiled_lines(mode_r, [P_l], [f_l], 0.0)
    rec("profiled FWM line reduces exactly to the delta line at zero width",
        "single line at the mode centre", "occupancy_from_line",
        md.occupancy_from_line([P_l], [f_l], mode_r), nu_delta, rel=1e-14)
    sig_p = 5e9
    conv = (P_l / (2 * H_PLANCK * f_l) /
            (np.sqrt(2 * np.pi * (mode_r.sigma ** 2 + sig_p ** 2))))
    rec("profiled line versus the analytic Gaussian convolution",
        "5 GHz rms profile, unit transfer",
        "N(0; sigma_mode^2 + sigma_profile^2)", float(conv),
        md.occupancy_from_profiled_lines(mode_r, [P_l], [f_l], sig_p), rel=1e-9,
        note="the 1/(h f) factor is frozen at f_l in the analytic reference")

    # -------------------------------------------------------------------- FWM
    atten = cf.build_attenuation(base)
    disp = at.Dispersion(beta2_s2_m=-21.7e-27, beta3_s3_m=0.12e-39, n_group=1.468)
    small = dpl.Deployment(freqs_hz=np.array([193.0e12, 193.1e12, 193.2e12]),
                           launch_w=np.full(3, dbm_to_w(-5.0)))
    pr3 = fw.enumerate_products(small, disp, atten, 1e4, 1.3e-3, quad_check=9)
    n_expect = sum(1 for i in range(3) for j in range(i, 3) for k in range(3)
                   if k != i and k != j)
    rec("small-triplet enumeration count and degeneracy convention", "3 channels",
        "combinatorial count with i<=j, k not in {i,j}", float(n_expect),
        float(pr3.n_triplets), tol=0.0)
    rec("FWM lossy amplitude: closed form vs resolved quadrature (3 channels)",
        "9 sampled triplets", "direct z quadrature, 300 points per phase cycle",
        0.0, float(pr3.audit["closed_vs_quadrature_max_rel"]), tol=1e-4)
    prf = fw.enumerate_products(dep0, disp, atten, 1e4, 1.3e-3, quad_check=25)
    rec("FWM lossy amplitude: closed form vs resolved quadrature (35 channels)",
        "25 sampled triplets", "direct z quadrature, 300 points per phase cycle",
        0.0, float(prf.audit["closed_vs_quadrature_max_rel"]), tol=1e-4)
    # --- independent low-conversion benchmarks (Methods: four-wave mixing)
    # These do NOT call the production prefactor: the expected powers are the
    # textbook seeded-FWM results for the declared scalar low-conversion model,
    #   degenerate    P_out = gamma^2 P_p^2 P_s L^2,
    #   nondegenerate P_out = 4 gamma^2 P_i P_j P_k L^2,
    # evaluated lossless and phase matched so the longitudinal integral is L.
    class _Lossless:
        valid_nm = (1000.0, 2000.0)

        def a_per_m(self, lam_nm, allow_extrapolation=False):
            return np.zeros_like(np.asarray(lam_nm, dtype=float))

    flat = at.Dispersion(beta2_s2_m=0.0, beta3_s3_m=0.0, n_group=1.468)
    lossless = _Lossless()
    g_bench, L_bench = 1.3e-3, 1.0e4
    Pp, Ps = 1.0e-3, 2.0e-4
    two = dpl.Deployment(freqs_hz=np.array([193.0e12, 193.1e12]),
                         launch_w=np.array([Pp, Ps]))
    pb = fw.enumerate_products(two, flat, lossless, L_bench, g_bench)
    i_deg = int(np.argmin(np.abs(pb.freq_hz - (2 * 193.0e12 - 193.1e12))))
    rec("FWM degenerate benchmark: lossless, phase matched",
        f"P_pump={Pp:g} W, P_seed={Ps:g} W, L={L_bench/1e3:g} km",
        "independent analytic P = gamma^2 P_p^2 P_s L^2",
        g_bench ** 2 * Pp ** 2 * Ps * L_bench ** 2,
        float(pb.power_incoherent_w[i_deg]), rel=1e-8,
        note="does not use the production prefactor routine; the residual is "
             "floating-point cancellation in delta-beta (beta ~ 6e6 rad/m must "
             "cancel to zero for a linear beta(f))")
    P1, P2, P3 = 1.0e-3, 7.0e-4, 3.0e-4
    three = dpl.Deployment(freqs_hz=np.array([193.0e12, 193.1e12, 193.3e12]),
                           launch_w=np.array([P1, P2, P3]))
    pn = fw.enumerate_products(three, flat, lossless, L_bench, g_bench)
    f_nd = 193.0e12 + 193.1e12 - 193.3e12
    i_nd = int(np.argmin(np.abs(pn.freq_hz - f_nd)))
    n_contrib = sum(1 for i in range(3) for j in range(i, 3) for k in range(3)
                    if k not in (i, j) and
                    abs(three.freqs_hz[i] + three.freqs_hz[j]
                        - three.freqs_hz[k] - f_nd) < 1.0)
    rec("only one unordered triplet reaches the nondegenerate benchmark line",
        "193.0 + 193.1 - 193.3 THz", "combinatorial count", 1.0,
        float(n_contrib), abs_tol=0.0)
    rec("FWM nondegenerate benchmark: lossless, phase matched",
        f"P=({P1:g},{P2:g},{P3:g}) W, L={L_bench/1e3:g} km",
        "independent analytic P = 4 gamma^2 P_i P_j P_k L^2",
        4.0 * g_bench ** 2 * P1 * P2 * P3 * L_bench ** 2,
        float(pn.power_incoherent_w[i_nd]), rel=1e-8,
        note="does not use the production prefactor routine; same delta-beta "
             "cancellation floor")
    pb2 = fw.enumerate_products(two, flat, lossless, 2 * L_bench, g_bench)
    rec("FWM scales as L^2 in the short phase-matched limit", "L and 2L",
        "analytic ratio", 4.0,
        float(pb2.power_incoherent_w[int(np.argmin(np.abs(
            pb2.freq_hz - (2 * 193.0e12 - 193.1e12))))]
            / pb.power_incoherent_w[i_deg]), rel=1e-8)
    pb3 = fw.enumerate_products(two, flat, lossless, L_bench, 2 * g_bench)
    rec("FWM scales as gamma^2", "gamma and 2 gamma", "analytic ratio", 4.0,
        float(pb3.power_incoherent_w[int(np.argmin(np.abs(
            pb3.freq_hz - (2 * 193.0e12 - 193.1e12))))]
            / pb.power_incoherent_w[i_deg]), rel=1e-12)
    two_x2 = dpl.Deployment(freqs_hz=two.freqs_hz, launch_w=2 * two.launch_w)
    pb4 = fw.enumerate_products(two_x2, flat, lossless, L_bench, g_bench)
    rec("FWM scales as the cube of a common launch-power scaling",
        "all powers x2", "analytic ratio", 8.0,
        float(pb4.power_incoherent_w[int(np.argmin(np.abs(
            pb4.freq_hz - (2 * 193.0e12 - 193.1e12))))]
            / pb.power_incoherent_w[i_deg]), rel=1e-12)
    # Fault injection: the earlier convention (d = 3, 6 with the conventional
    # gamma) must be REJECTED by the same benchmark.
    pbad = fw.enumerate_products(two, flat, lossless, L_bench, 3.0 * g_bench,
                                 convention="gamma_over_three")
    bad_ratio = float(pbad.power_incoherent_w[int(np.argmin(np.abs(
        pbad.freq_hz - (2 * 193.0e12 - 193.1e12))))]
        / (g_bench ** 2 * Pp ** 2 * Ps * L_bench ** 2))
    rec("fault injection: the earlier d=3,6 with conventional gamma is rejected",
        "same benchmark, buggy normalization",
        "ratio to the analytic degenerate benchmark", 9.0, bad_ratio, rel=1e-8,
        note="the benchmark REJECTS the buggy model: its ratio to the analytic "
             "result is 9, so the 1e-8 relative check on the corrected model "
             "cannot pass with the old normalization")

    prf2 = fw.enumerate_products(dep0, disp, atten, 1e4, 1.3e-3)
    rec("generated FWM frequencies are fixed by the classical grid",
        "independent of receiver tuning", "two enumerations", 0.0,
        float(np.abs(prf.freq_hz - prf2.freq_hz).max()), tol=0.0)
    coh = float(np.sum(prf.power_coherent_w) / np.sum(prf.power_incoherent_w))
    report("coherent versus incoherent product summation",
           "35-channel grid, products grouped by frequency",
           "|sum A|^2 versus sum |A|^2", 1.0, coh,
           f"coherent/incoherent = {coh:.4f}; the principal model uses the mutually "
           f"incoherent sum, which is an explicit assumption")
    report("FWM directional reception", "pass 2 receives at terminal A",
           "forward generation only", 0.0, 0.0,
           "forward-generated products are collected at passes 1 and 3 only; "
           "backward generation, reflection and backscatter are NOT modeled")

    # ---------------------------------------------------------------- leakage
    S_rem = float(lk.incident_spectrum(np.array([f0]), dep0, atten, 1e4, False,
                                       25e9, 2.5e-14, 1.0)[0])
    S_loc = float(lk.incident_spectrum(np.array([f0]), dep0, atten, 1e4, True,
                                       25e9, 2.5e-14, 1.0)[0])
    lam_c = C_LIGHT / dep0.freqs_hz * 1e9
    span = np.exp(-atten.a_per_m(lam_c) * 1e4)
    shape = np.array(lk.modulation_lineshape(f0, dep0.freqs_hz, 25e9)) + 2.5e-14
    w = dep0.launch_w * shape
    expected_ratio = float(np.sum(w) / np.sum(w * span))
    rec("local leakage path does not traverse the span", "10 km",
        "spectrum-weighted local/remote incident ratio", expected_ratio,
        S_loc / S_rem, tol=1e-12,
        note=f"the local path is {expected_ratio:.4f}x the remote one; the legacy "
             f"code applied exp(-aL) to every path")
    rec("zero span length makes the two leakage paths identical", "L = 0",
        "local vs remote incident spectra", 1.0,
        float(lk.incident_spectrum(np.array([f0]), dep0, atten, 0.0, True, 25e9,
                                   2.5e-14, 1.0)[0] /
              lk.incident_spectrum(np.array([f0]), dep0, atten, 0.0, False, 25e9,
                                   2.5e-14, 1.0)[0]), tol=1e-15)
    rec("path isolation is applied exactly once", "110 dB",
        "T_sys(f0) = T_q(f0) 10^(-110/10)", 1e-11,
        float(lk.system_transfer(np.array([f0]), filt, 110.0)[0]), tol=1e-16)
    rec("modeled leakage spectrum is finite outside the declared domain",
        "1400 nm", "T_q outside 1500-1600 nm", 0.0, float(filt(nm_to_hz(1400.0))),
        tol=0.0)

    # -------------------------------------------------------------------- XPM
    a_ci = atten.a_per_m(lam_c)
    d_co = disp.walkoff_co(dep0.freqs_hz, f0)
    d_ct = disp.walkoff_counter(dep0.freqs_hz, f0)
    args = (1e4, 1.3e-3, 1.0, 1.0, 25e9)
    rec("zero differential coupling gives zero phase variance", "kappa = 0",
        "sigma2_closed", 0.0,
        xp.sigma2_closed(dep0, a_ci, d_co, 1e4, 1.3e-3, 0.0, 1.0, 25e9), tol=0.0)
    rec("zero residual power fluctuation gives zero phase variance",
        "modulation index = 0", "sigma2_closed", 0.0,
        xp.sigma2_closed(dep0, a_ci, d_co, 1e4, 1.3e-3, 1.0, 0.0, 25e9), tol=0.0)
    s_cl = xp.sigma2_closed(dep0, a_ci, d_co, *args)
    s_sp, resolved = xp.sigma2_spectral(dep0, a_ci, d_co, *args, n_F=400001)
    s_cv = xp.sigma2_covariance(dep0, a_ci, d_co, *args, n_z=4001)
    rec("XPM closed form vs frequency-domain PSD quadrature (co-propagating)",
        "35 channels, 10 km, 25 GHz band", "direct integral of S|H|^2", s_sp, s_cl,
        cutoff="n_F=400001", tol=1e-6, note=f"quadrature resolved: {resolved}")
    rec("XPM closed form vs position-domain covariance double integral (co)",
        "35 channels, 10 km", "double integral of the power covariance", s_cv,
        s_cl, cutoff="n_z=4001", tol=1e-4)
    s_ct = xp.sigma2_closed(dep0, a_ci, d_ct, *args)
    _, resolved_ct = xp.sigma2_spectral(dep0, a_ci, d_ct, *args, n_F=400001)
    need = float(np.max(xp.resolution_requirement(d_ct, 1e4, 25e9)))
    report("XPM counter-propagating: both reference routes are unresolvable",
           "35 channels, 10 km, 25 GHz band",
           "frequency quadrature / position covariance", need, 400001.0,
           f"resolved={resolved_ct}; ~{need:.2e} quadrature points would be needed, "
           f"so the closed form is validated against its analytic large-walk-off "
           f"limit instead", cutoff="n_F=400001")
    K0 = (1 - np.exp(-2 * a_ci * 1e4)) / (2 * a_ci)
    asym = float(4 * (1.3e-3) ** 2 * 1.0 ** 2 *
                 np.sum((1.0 * dep0.launch_w) ** 2 * K0 / (2 * 25e9 * d_ct)))
    rec("XPM counter-propagating closed form vs analytic large-walk-off limit",
        "35 channels, 10 km", "K(0)/(2 B d) asymptote", asym, s_ct, tol=1e-4)
    leg = xp.sigma2_legacy_capped(dep0, a_ci, d_co, *args)
    report("legacy capped-walk-off approximation (labelled, not used)",
           "35 channels, 10 km, co-propagating", "ratio to the exact closed form",
           s_cl, leg, f"legacy/exact = {leg/s_cl:.4f}; the legacy asymptote is "
                      f"correct but the crossover to the CW limit is not")
    emp, ana = xp.coherence_check(0.4, 400000, seed=7)
    rec("Gaussian phase coherence equals exp(-sigma^2/2)", "sigma^2 = 0.4",
        "Monte Carlo over 4e5 phases", ana, emp, tol=3e-3)
    rec("coherence-to-z conversion round trip", "z = 1e-5",
        "z = (1-exp(-sigma^2/2))/2", 1e-5,
        float(lp.z_from_sigma2(lp.sigma2_from_z(np.array(1e-5)))), tol=1e-15)

    # ------------------------------------------------------------- data checks
    try:
        atten.db_per_km(1200.0); ok_ex = False
    except ValueError:
        ok_ex = True
    rec("unsupported extrapolation is rejected", "1200 nm vs the tabulated range",
        "TabulatedAttenuation.db_per_km", 1.0, float(ok_ex), tol=0.0)
    prov = atten.rows()
    n_spec = sum(1 for r in prov if r["provenance_category"] == "manufacturer_specification")
    in_band = [r for r in prov if 1530.0 <= r["wavelength_nm"] <= 1565.0
               and r["provenance_category"] == "manufacturer_specification"]
    report("attenuation provenance inside the quantum band",
           "1530-1565 nm", "per-row provenance of the supplied CSV",
           float(n_spec), float(len(in_band)),
           f"{len(in_band)} datasheet anchor(s) lie inside 1530-1565 nm out of "
           f"{n_spec} in the file; the rest of the band is interpolated, so it is an "
           f"assumed curve rather than a measured or pointwise-bounded spectrum")

    # ------------------------------------------ optical reduction limit cases
    r_pure = oc.accepted_optical(0.37, 0.0)
    rec("pure loss gives s = T", "T=0.37, nu=0", "closed form", 0.37,
        float(r_pure.s), tol=1e-15)
    rec("pure loss gives r = 0", "T=0.37, nu=0", "closed form", 0.0,
        float(r_pure.r), tol=1e-15)
    rec("identity channel", "T=1, nu=0", "closed form", 1.0,
        float(oc.accepted_optical(1.0, 0.0).s), tol=1e-15)
    rec("complete signal loss with background gives r = 1", "T=0, nu=1e-3",
        "closed form", 1.0, float(oc.accepted_optical(0.0, 1e-3).r), tol=1e-15)
    r_gt = oc.accepted_optical(0.05, 0.05)
    rec("s can exceed T through background replacement", "T=0.05, nu=0.05",
        "closed form", 1.0, 1.0 if float(r_gt.s) > 0.05 else 0.0, tol=0.0,
        note=f"s = {float(r_gt.s):.6f} > T, so T^n is not a general upper bound")
    rec("D = 0 leaves the conditional quantities undefined", "T=0, nu=0",
        "closed form", 1.0, float(not bool(oc.accepted_optical(0.0, 0.0).defined)),
        tol=0.0)

    # ------------------------------------------------- Stage B dilation grid
    ov = cfg["optical_dilation"]
    nu_grid = [0.0] + list(np.logspace(ov["nu_log_min"], ov["nu_log_max"],
                                       ov["nu_points"]))
    rng = np.random.default_rng(ov["random_seed"])
    pts = [(T, nu) for T in ov["T_grid"] for nu in nu_grid] + \
          [(float(rng.uniform(0.02, 0.9)),
            float(10 ** rng.uniform(ov["nu_log_min"], ov["nu_log_max"])))
           for _ in range(ov["random_interior_points"])]
    states = {"|0>": np.array([[1, 0], [0, 0]], complex),
              "|1>": np.array([[0, 0], [0, 1]], complex),
              "|+>": 0.5 * np.array([[1, 1], [1, 1]], complex),
              "|->": 0.5 * np.array([[1, -1], [-1, 1]], complex),
              "|+i>": 0.5 * np.array([[1, -1j], [1j, 1]], complex),
              "|-i>": 0.5 * np.array([[1, 1j], [-1j, 1]], complex)}
    val_rows = []
    worst_abs = worst_cond = worst_spread = 0.0
    for T, nu in pts:
        res = oc.accepted_optical(T, nu)
        s_a, c0_a, c1_a = float(res.s), float(res.c0), float(res.c1)
        J_num, Ech, info = od.accepted_choi_numeric(T, nu, tail_tol=1e-16)
        J_ana = oc.choi_from_weights(c0_a, c1_a)
        tn = float(np.linalg.svd(J_num - J_ana, compute_uv=False).sum())
        s_n = float(np.real(np.trace(J_num)))
        accs, st_err = [], 0.0
        for _, rho in states.items():
            out = od.accepted_map_numeric(T, nu, rho, Ech=Ech)
            accs.append(float(np.real(np.trace(out))))
            ref = c0_a * rho + c1_a * 0.5 * np.eye(2) * np.trace(rho)
            st_err = max(st_err, float(np.abs(out - ref).max()))
        spread = float(max(accs) - min(accs))
        ev = np.linalg.eigvalsh(J_num)
        tp = (float(np.abs((J_num[:2, :2] + J_num[2:, 2:]) / s_n - 0.5 * np.eye(2)).max())
              if s_n > 0 else np.nan)
        # TRUE normalized-channel comparison: the trace norm of the difference
        # of the two NORMALIZED Choi operators, not tn/s (brief Sec. 4.1).
        cond = (float(np.linalg.svd(J_num / s_n - J_ana / s_a,
                                    compute_uv=False).sum())
                if s_n > 0 and s_a > 0 else np.nan)
        cond_scaled = tn / s_n if s_n > 0 else np.nan
        worst_abs = max(worst_abs, tn); worst_spread = max(worst_spread, spread)
        if np.isfinite(cond):
            worst_cond = max(worst_cond, cond)
        c0_n = float(np.real(2.0 * J_num[0, 3]))     # <00|J|11> = c0/2
        c1_n = s_n - c0_n
        r_n = c1_n / s_n if s_n > 0 else np.nan
        val_rows.append(dict(T=T, nu=nu, s_analytic=s_a, s_numeric=s_n,
                             r_analytic=float(res.r), r_numeric=r_n,
                             c0_analytic=c0_a, c0_numeric=c0_n,
                             c1_analytic=c1_a, c1_numeric=c1_n,
                             choi_trace_norm_error=tn,
                             choi_conditional_error=cond,
                             choi_abs_error_over_s=cond_scaled,
                             max_state_error=st_err,
                             acceptance_spread=spread,
                             min_choi_eigenvalue=float(ev.min()),
                             trace_nonincrease_ok=bool(s_n <= 1 + 1e-12),
                             normalized_choi_tp_residual=tp,
                             env_cutoff_M=info["env_cutoff_M"],
                             discarded_tail=info["discarded_tail"],
                             two_env_tail_bound=info["two_env_tail_bound"]))
    man.add_output(io.write_csv(os.path.join(d, "optical_validation.csv"), val_rows))
    rec("accepted Choi: independent thermal dilation vs closed form",
        f"{len(pts)} (T,nu) points",
        "beam-splitter dilation with truncated thermal environments, exact within "
        "each photon-number sector", 0.0, worst_abs, cutoff="tail_tol=1e-16",
        abs_tol=ov["target_absolute_tolerance"],
        note="max Choi trace-norm error; truncation, floating-point residual and "
             "conditional normalization are reported as separate rows")
    rec("normalized-channel Choi error", f"{len(pts)} points",
        "|| J_num/s_num - J_formula/s_formula ||_1 (NOT the absolute error "
        "divided by s)", 0.0, worst_cond,
        abs_tol=ov["target_conditional_tolerance"])
    rec("acceptance is independent of the input state",
        "6 inputs including |+i>, |-i>", "max-min acceptance over the six inputs",
        0.0, worst_spread, tol=1e-12)
    rec("normalized Choi is trace preserving where s > 0", f"{len(pts)} points",
        "Tr_out J / s versus I/2", 0.0,
        float(np.nanmax([r["normalized_choi_tp_residual"] for r in val_rows])),
        tol=1e-12)
    rec("accepted Choi positivity", f"{len(pts)} points", "eigenvalues", 0.0,
        max(0.0, -min(r["min_choi_eigenvalue"] for r in val_rows)), tol=1e-12)
    rec("trace nonincrease", f"{len(pts)} points", "Tr J <= 1", 0.0,
        0.0 if all(r["trace_nonincrease_ok"] for r in val_rows) else 1.0, tol=0.0)
    rec("conditional mixing r from the dilation vs the closed form",
        f"{len(pts)} points", "r = c1/s read off the numerical Choi operator", 0.0,
        float(np.nanmax([abs(r["r_numeric"] - r["r_analytic"]) for r in val_rows
                         if np.isfinite(r["r_numeric"])])), tol=1e-12)

    # ------------------------------------------------ truncation convergence
    conv_rows = []
    for (T, nu) in [(0.9, 1e-1), (0.6, 1e-2), (0.2, 1e-3), (0.05, 1e-1)]:
        for tol in ov["thermal_tail_tolerances"]:
            J, _, info = od.accepted_choi_numeric(T, nu, tail_tol=tol)
            res = oc.accepted_optical(T, nu)
            Jf = oc.choi_from_weights(float(res.c0), float(res.c1))
            tn = float(np.linalg.svd(J - Jf, compute_uv=False).sum())
            conv_rows.append(dict(T=T, nu=nu, tail_tolerance=tol,
                                  env_cutoff_M=info["env_cutoff_M"],
                                  discarded_tail=info["discarded_tail"],
                                  two_env_tail_bound=info["two_env_tail_bound"],
                                  choi_trace_norm_error=tn,
                                  conditional_error=tn / float(res.s)))
    man.add_output(io.write_csv(os.path.join(d, "optical_convergence.csv"), conv_rows))

    # ------------------------------ three explicit traversals, final acceptance
    comp_rows = []
    for eta, nup, dim in [([0.55] * 3, [3e-4] * 3, 7),
                          ([0.8, 0.7, 0.6], [1e-4, 2e-4, 3e-4], 7),
                          ([0.9] * 3, [1e-3] * 3, 9),
                          ([0.6] * 3, [1e-3, 0.0, 1e-3], 8)]:
        T, nu, prop = oc.compose_passes(eta, nup)
        J3, _ = od.three_traversal_accepted(eta, nup, dim=dim)
        res = oc.accepted_optical(T, nu)
        Jf = oc.choi_from_weights(float(res.c0), float(res.c1))
        err = float(np.linalg.svd(J3 - Jf, compute_uv=False).sum())
        comp_rows.append(dict(eta=str(eta), nu_pass=str(nup), fock_cutoff=dim, T=T,
                              nu_effective=nu,
                              nu_contributions_propagated=str([float(x) for x in prop]),
                              s_three_pass=float(np.real(np.trace(J3))),
                              s_effective=float(res.s), choi_trace_norm_error=err))
        rec("three explicit traversals versus one effective (T,nu) channel",
            f"eta={eta}, nu={nup}",
            "Fock propagation retaining all intermediate multiphoton sectors, "
            "projection only at the end", 0.0, err, cutoff=f"dim={dim}", tol=1e-12)
    man.add_output(io.write_csv(os.path.join(d, "optical_composition.csv"), comp_rows))
    rec("earlier background is attenuated by later traversals",
        "eta=(0.5,0.5,0.5), nu=(1,0,0)", "nu = eta3 eta2 nu1", 0.25,
        float(oc.compose_passes([0.5] * 3, [1.0, 0.0, 0.0])[1]), tol=1e-15)
    rec("terminal background is not attenuated", "nu=(0,0,1)", "nu = nu3", 1.0,
        float(oc.compose_passes([0.5] * 3, [0.0, 0.0, 1.0])[1]), tol=1e-15)

    man.add_output(io.write_csv(os.path.join(d, "validation_table_s2_optics.csv"), rows))
    man.add_output(io.write_csv(os.path.join(d, "migration_record.csv"),
                                lg.migration_rows()))
    man.add_output(io.write_csv(os.path.join(d, "attenuation_provenance.csv"), prov))
    man.add_input(os.path.join(io.ROOT, "data/raw/attenuation_smf28.csv"),
                  "mixed_per_row", "legacy data_templates/attenuation_smf28.csv")
    man.save(os.path.join(d, "run_manifest_optics.json"))

    checks = [r for r in rows if r["row_kind"] == "assert"]
    diags = [r for r in rows if r["row_kind"] != "assert"]
    nfail = sum(1 for r in checks if r["pass_fail_and_reason"].startswith("FAIL"))
    print(f"\nStage B complete: {len(checks)} asserted checks ({nfail} failures), "
          f"{len(diags)} report-only diagnostics, {len(val_rows)} dilation sweep "
          f"points, {len(conv_rows)} convergence points")
    for r in rows:
        tag = r["pass_fail_and_reason"][:4]
        ae = r["absolute_error"]
        aes = "  n/a   " if not np.isfinite(ae) else f"{ae:.3e}"
        print(f"  [{tag:4s}] {r['validation_case']:<64s} err={aes}")
    return nfail


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
