"""Assemble the per-traversal optical quantities (eta_m, nu_m, sigma2_m) for a
candidate quantum frequency under a frozen classical deployment.

Reference planes follow `deployment.ComponentLedger`:
  eta_m   = eta_tx * exp(-a(lambda_q) L) * eta_rx * <T_q>_mode
  in-span background sees eta_rx * T_q(f) inside the occupancy integral
  leakage sees its own path transfer T_sys(f), applied exactly once
"""
from dataclasses import dataclass, field
import numpy as np

import raman as rm
import fwm as fw
import leakage as lk
import xpm as xp
from modes import GaussianMode, SuperGaussianFilter, occupancy_from_spectrum, \
    occupancy_from_line, occupancy_from_profiled_lines, \
    mode_overlap_transmission, power_average_transmission
from units import C_LIGHT


@dataclass
class Scenario:
    """Defaults mirror configs/illustrative_baseline.yaml, which is the
    authoritative source; every run builds this object through src/config.py."""
    dep: object
    atten: object
    disp: object
    length_m: float = 10_000.0
    eta_tx: float = np.sqrt(0.82)
    eta_rx: float = np.sqrt(0.82)
    mode_fwhm_hz: float = 10e9
    filter_fwhm_hz: float = 25e9
    filter_order: int = 4
    filter_isolation_db: float = 110.0        # receive filter out-of-band floor
    # Receiver convention (Sec. 3.1 of the Methods).
    #   "fixed_mode_overlap": the accepted input and output modes are the SAME
    #       fixed normalized mode psi, and the signal transmission is the modal
    #       overlap |int |psi|^2 t(f) df|^2 through the actual field transfer
    #       t = sqrt(T_q).  Noise is projected into that same fixed mode.
    #   "power_average": the legacy int |psi|^2 T_q df.  Retained only as a
    #       declared sensitivity scenario; it is a power average, not a modal
    #       projection, so it is not mixed with the overlap in any main result.
    # The filter is taken to have no spectral phase, so t is real and positive;
    # propagation phase is assumed compensated or absorbed into the mode.
    receiver_mode_convention: str = "fixed_mode_overlap"
    modeled_domain_nm: tuple = (1500.0, 1600.0)
    e0_per_pass: float = 0.002
    # Raman
    raman_model: str = "multi_gaussian_v1"
    raman_scale: float = 1.0
    temperature_k: float = 300.0
    # FWM
    gamma_per_w_km: float = 1.3
    enable_fwm: bool = True
    fwm_convention: str = "conventional_gamma"
    # Prescribed rms spectral width of every generated FWM component.  0 keeps
    # the analytic narrow-line (delta) limit.
    fwm_profile_rms_hz: float = 0.0
    # leakage
    mod_rms_hz: float = 25e9
    tx_broadband_floor_per_hz: float = 2.5e-14   # fraction of carrier power per Hz
    # Path isolation from classical LAUNCH to the quantum receiver's FILTER INPUT.
    # It contains the classical transmit-side filtering, mux and demux rejection
    # but NOT the final quantum filter T_q, which is applied separately once.
    leak_isolation_remote_db: float = 110.0
    leak_isolation_local_db: float = 90.0
    enable_leakage: bool = True
    # XPM
    xpm_kappa: float = 1.0
    xpm_rel_fluct: float = 1.0
    xpm_band_hz: float = 25e9
    enable_xpm: bool = True
    # numerics
    gh_order: int = 48
    mode_line_window_sigma: float = 12.0

    _fwm_cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ helpers
    @property
    def modeled_domain_hz(self):
        lo_nm, hi_nm = self.modeled_domain_nm
        return (C_LIGHT / (hi_nm * 1e-9), C_LIGHT / (lo_nm * 1e-9))

    @property
    def gamma_per_w_m(self):
        return self.gamma_per_w_km / 1e3

    def fwm_products(self):
        key = (self.length_m, float(np.sum(self.dep.launch_w)), self.gamma_per_w_km,
               self.fwm_convention)
        if key not in self._fwm_cache:
            self._fwm_cache[key] = fw.enumerate_products(
                self.dep, self.disp, self.atten, self.length_m,
                self.gamma_per_w_m, quad_check=25,
                convention=self.fwm_convention)
        return self._fwm_cache[key]

    def a_classical(self):
        lam_nm = C_LIGHT / self.dep.freqs_hz * 1e9
        return self.atten.a_per_m(lam_nm)


def evaluate_candidate(sc, f_q, want_components=True):
    """Return a dict of per-pass optical quantities for the candidate frequency."""
    lam_q_nm = C_LIGHT / f_q * 1e9
    a_q = float(sc.atten.a_per_m(lam_q_nm))
    a_ci = sc.a_classical()
    mode = GaussianMode(f_q, sc.mode_fwhm_hz)
    filt = SuperGaussianFilter(f_q, sc.filter_fwhm_hz, sc.filter_order,
                               sc.filter_isolation_db, sc.modeled_domain_hz)

    # Signal: one receiver convention, applied ONCE per traversal.
    if sc.receiver_mode_convention == "fixed_mode_overlap":
        T_filt_signal = mode_overlap_transmission(
            mode, lambda f: np.sqrt(filt(f)), sc.gh_order)
    elif sc.receiver_mode_convention == "power_average":
        T_filt_signal = power_average_transmission(mode, filt, sc.gh_order)
    else:
        raise ValueError(f"unknown receiver convention "
                         f"{sc.receiver_mode_convention!r}")
    span = np.exp(-a_q * sc.length_m)
    eta_m = sc.eta_tx * span * sc.eta_rx * T_filt_signal

    out = {"wavelength_nm": lam_q_nm, "frequency_hz": f_q,
           "a_q_per_m": a_q, "T_filter_signal": T_filt_signal,
           "receiver_mode_convention": sc.receiver_mode_convention,
           "eta": np.full(3, eta_m), "span_transmission": span}

    nu_R = np.zeros(3)
    nu_F = np.zeros(3)
    nu_L = np.zeros(3)
    sig2 = np.zeros(3)

    prods = sc.fwm_products() if sc.enable_fwm else None
    if prods is not None:
        win = sc.mode_line_window_sigma * (mode.sigma + float(sc.fwm_profile_rms_hz))
        lo, hi = np.searchsorted(prods.freq_hz, [f_q - win, f_q + win])
        fl = prods.freq_hz[lo:hi]
        pl = prods.power_incoherent_w[lo:hi]
        pl_coh = prods.power_coherent_w[lo:hi]
    for m in range(3):
        co = sc.dep.copropagating(m)
        local = sc.dep.local_transmitter(m)

        # --- Raman: generated in span, then eta_rx and the filter
        def S_raman(f, co=co):
            a_obs = sc.atten.a_per_m(C_LIGHT / f * 1e9, allow_extrapolation=True)
            S = rm.span_spectrum(f, sc.dep, a_obs, a_ci, sc.length_m, co,
                                 sc.raman_model, sc.temperature_k, sc.raman_scale)
            return S * sc.eta_rx * filt(f)
        nu_R[m] = occupancy_from_spectrum(mode, S_raman, sc.gh_order)

        # --- FWM: forward generation only, collected only on forward passes
        if prods is not None and sc.dep.quantum_directions[m] == sc.dep.direction:
            nu_F[m] = occupancy_from_profiled_lines(
                mode, pl, fl, sc.fwm_profile_rms_hz,
                transfer_fn=lambda f: sc.eta_rx * filt(f), order=sc.gh_order)
        else:
            nu_F[m] = 0.0

        # --- leakage: path-specific
        if sc.enable_leakage:
            iso_db = sc.leak_isolation_local_db if local else sc.leak_isolation_remote_db

            def S_leak(f, local=local, iso_db=iso_db):
                S_inc = lk.incident_spectrum(f, sc.dep, sc.atten, sc.length_m,
                                             local, sc.mod_rms_hz,
                                             sc.tx_broadband_floor_per_hz,
                                             local_tap_transmission=1.0)
                return S_inc * lk.system_transfer(f, filt, iso_db) * (
                    1.0 if local else sc.eta_rx)
            nu_L[m] = occupancy_from_spectrum(mode, S_leak, sc.gh_order)

        # --- residual differential XPM
        if sc.enable_xpm:
            if co:
                d = sc.disp.walkoff_co(sc.dep.freqs_hz, f_q)
            else:
                d = sc.disp.walkoff_counter(sc.dep.freqs_hz, f_q)
            sig2[m] = xp.sigma2_closed(sc.dep, a_ci, d, sc.length_m,
                                       sc.gamma_per_w_m, sc.xpm_kappa,
                                       sc.xpm_rel_fluct, sc.xpm_band_hz)

    out.update({"nu_R": nu_R, "nu_FWM": nu_F, "nu_leak": nu_L,
                "nu": nu_R + nu_F + nu_L, "sigma2_phi": sig2})
    if want_components and prods is not None:
        out["fwm_lines_in_window"] = int(len(fl))
        out["fwm_coherent_over_incoherent"] = (
            float(np.sum(pl_coh) / np.sum(pl)) if len(pl) and np.sum(pl) > 0 else np.nan)
    return out
