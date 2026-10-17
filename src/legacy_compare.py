"""Quantitative record of the required migration from the legacy scripts
(the revision notes in CHANGES.md).  Each row is checked against the extracted legacy code."""
import numpy as np
from units import C_LIGHT, H_PLANCK, K_BOLTZ, nm_to_hz
import raman as rm

# Legacy constants, transcribed from legacy/code/single_fiber_figures_colab_v2.py
LEGACY_RAMAN_RHO = 2.85e-9 / 1000.0 / (C_LIGHT / (1550e-9) ** 2 * 1e-9)
LEGACY_PEAK_THZ = 13.2
LEGACY_TEMPERATURE_K = 300.0


def legacy_raman_rho(delta_f_hz, thermal=True):
    """The legacy single-peak shape, with RAMAN_RHO_LIT assigned to the PEAK."""
    d_thz = np.asarray(delta_f_hz, dtype=float) / 1e12
    det = np.abs(d_thz)
    x = np.maximum(det / LEGACY_PEAK_THZ, 1e-15)
    shape = np.where(det == 0.0, 0.0, x ** 2 * np.exp(2.0 * (1.0 - x)))
    if thermal:
        ratio = np.exp(-H_PLANCK * det * 1e12 / (K_BOLTZ * LEGACY_TEMPERATURE_K))
        side = np.where(d_thz >= 0.0, 1.0, ratio)
    else:
        side = np.where(d_thz >= 0.0, 1.0, 0.60)
    return LEGACY_RAMAN_RHO * side * shape


def raman_recalibration_factor(delta_f_hz, model="multi_gaussian_v1"):
    """Ratio of the recalibrated coefficient to the legacy one at a detuning."""
    new = rm.calibrated_rho(delta_f_hz, model=model)
    old = legacy_raman_rho(delta_f_hz)
    return np.asarray(new) / np.asarray(old)


def migration_rows():
    """The Section 3 table with the numerical consequence of each item."""
    dcal = rm.QI_DELTA_F_HZ
    r_cal = float(raman_recalibration_factor(dcal))
    r_1 = float(raman_recalibration_factor(1e12))
    r_4 = float(raman_recalibration_factor(4e12))
    return [
        dict(legacy_item="evaluate(): power -> gate counts, qbg = 1-exp(-mu_bg)",
             problem="converts an integrated multimode gate count into a qubit error",
             action_taken="replaced by mode occupancy nu, thermal-channel composition, "
                          "and a single FINAL acceptance",
             numerical_consequence="mu_gate/nu = B*tau_g temporal modes; measured in "
                                   "validate_optics.py"),
        dict(legacy_item="p_survival = eta3**css_n ; p_use",
             problem="all-signal-survival model; not the acceptance of the revised receiver",
             action_taken="retained only as a named diagnostic; the upper bound is s**n",
             numerical_consequence="s > T whenever nu > 0 and T is small, so T**n is not "
                                   "an upper bound"),
        dict(legacy_item="message_sim.py binomial weight <= 1 test",
             problem="samples the same binomial tail it is meant to validate",
             action_taken="replaced by explicit syndrome recovery: 4**7 Pauli enumeration "
                          "and an 8-qubit density-matrix calculation",
             numerical_consequence="exact Fe exceeds C_7,1(q) because some weight>1 "
                                   "histories are also corrected"),
        dict(legacy_item="RAMAN_RHO_LIT assigned to raman_peak_per_m_hz",
             problem="a coefficient reported at a -0.60 THz anti-Stokes separation is "
                     "used as the 13.2 THz Stokes-peak normalization",
             action_taken="renormalized at the measurement's actual separation: "
                          "rho(df) = rho_meas g(df)/g(df_meas)",
             numerical_consequence=f"rho_new/rho_legacy = {r_cal:.1f}x at the calibration "
                                   f"point, {r_1:.1f}x at +1 THz, {r_4:.1f}x at +4 THz"),
        dict(legacy_item="raman_power(): rho * filter_fwhm_ghz, returns total power",
             problem="FWHM used as the integrated noise bandwidth; no spectrum returned",
             action_taken="span_spectrum() returns a PSD; occupancy comes from the "
                          "accepted-mode overlap; B_eq computed separately",
             numerical_consequence="B_eq differs from the 25 GHz FWHM; reported in "
                                   "validate_optics.py"),
        dict(legacy_item="fwm_products(): L_eff-style factor times separate sinc^2",
             problem="product approximation to the squared lossy amplitude integral",
             action_taken="exact lossy amplitude integral with signed wavevectors, "
                          "cross-checked against phase-resolved quadrature",
             numerical_consequence="closed form vs resolved quadrature agree to <1e-4 "
                                   "relative"),
        dict(legacy_item="fwm_pass_scale = (1,1,1)",
             problem="reuses forward-generated products on the return receiver",
             action_taken="forward generation is collected only at the forward terminal; "
                          "passes 1 and 3 receive it, pass 2 receives none",
             numerical_consequence="nu_FWM_2 = 0 by construction; backward generation is "
                                   "declared NOT modeled"),
        dict(legacy_item="leakage_power(): exp(-a L) on every path",
             problem="a remote-transmitter path applied to a local return receiver",
             action_taken="separate local and remote paths with separate isolation; the "
                          "local path does not traverse the span",
             numerical_consequence="local/remote incident ratio = 1/exp(-aL); the return "
                                   "pass leakage is larger than the legacy value"),
        dict(legacy_item="xpm_sigma2_theory(): min(walkoff, L_eff^2) cap",
             problem="capped walk-off approximation presented as first-principles",
             action_taken="exact closed form of int S|H|^2 dF, validated against a "
                          "frequency-domain quadrature and a position-domain covariance "
                          "double integral",
             numerical_consequence="legacy/exact ratio reported in validate_optics.py; "
                                   "the legacy asymptote is correct, the crossover is not"),
        dict(legacy_item="eta_receiver_mode = 0.35",
             problem="entered only the old background gate-count conversion",
             action_taken="NOT carried over; the receive plane is represented by eta_rx "
                          "plus the filter response, each applied once",
             numerical_consequence="removed; the component ledger records every plane"),
        dict(legacy_item="feasible_mask(): reserved window + guard bands",
             problem="conflates an empty deployment window with an imposed search restriction",
             action_taken="hardware domain, guard bands, and the reserved-window policy "
                          "are three separate masks with an exclusion reason per candidate",
             numerical_consequence="restrict_quantum_to_reserved_window is false in the "
                                   "principal run and true in a separate labelled run"),
        dict(legacy_item="classical_grid() regenerates the allocation",
             problem="the deployment can change when the search band changes",
             action_taken="the 35 frequencies are frozen from the fixed classical allocation and exported",
             numerical_consequence="identical classical_channels.csv across all scans"),
        dict(legacy_item="attenuation_smf28.csv presented as literature-sourced",
             problem="mixes datasheet MAXIMUM specs, author-placed points and extrapolations",
             action_taken="per-row provenance; datasheet anchors marked as bounds; "
                          "extrapolated rows marked and excluded from principal claims",
             numerical_consequence="only 1550 nm is a datasheet anchor inside 1530-1565 nm; "
                                   "the rest of the quantum band is interpolated"),
        dict(legacy_item="README: 'all defaults are literature-sourced'",
             problem="too strong",
             action_taken="replaced by per-row provenance categories in provenance.csv",
             numerical_consequence="most optical inputs are illustrative_assumption"),
    ]
