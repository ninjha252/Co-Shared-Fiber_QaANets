"""Configuration loading and resolution into runtime objects.

A configuration file is the single source of truth for a run: the scripts read
it, build the deployment, attenuation, dispersion and scenario objects from it,
and export the same dictionary into the run manifest and the parameter table.
Runtime overrides are written back into the dictionary before anything is
computed, so an exported table always describes the run that happened.
"""
import os
import numpy as np
import yaml

import deployment as dp
import attenuation as at
import optical_link as ol
from units import dbm_to_w, nm_to_hz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "configs")


def load(name):
    path = name if os.path.isabs(name) else os.path.join(CONFIGS, name)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = path
    return cfg


def build_deployment(cfg, launch_dbm=None):
    d = cfg["deployment"]
    f = np.array(d["classical_frequencies_ghz"], dtype=float) * 1e9
    p = dbm_to_w(d["classical_launch_dbm_per_channel"] if launch_dbm is None else launch_dbm)
    return dp.Deployment(freqs_hz=f, launch_w=np.full(f.shape, p),
                         direction=d["classical_direction"],
                         quantum_directions=tuple(d["quantum_directions"]),
                         classical_tx_terminal=d["classical_tx_terminal"],
                         quantum_rx_terminal=tuple(d["quantum_rx_terminal"]))


def build_attenuation(cfg, source=None):
    o = cfg["optics"]
    src = source or o.get("attenuation_source", "tabulated_smf28")
    if src == "tabulated_smf28":
        path = os.path.join(ROOT, o.get("attenuation_csv",
                                        "data/raw/attenuation_smf28.csv"))
        return at.TabulatedAttenuation(path, valid_nm=tuple(o["attenuation_valid_nm"]))
    if src == "analytic_three_term":
        return at.AttenuationModel(anchor_nm=o["attenuation_anchor_nm"],
                                   anchor_db_km=o["attenuation_anchor_db_km"],
                                   valid_nm=tuple(o["attenuation_valid_nm"]))
    raise ValueError(f"unknown attenuation source {src!r}")


def build_scenario(cfg, length_km=None, launch_dbm=None, overrides=None):
    o = cfg["optics"]; lk = cfg["link"]; num = cfg["numerics"]
    dep = build_deployment(cfg, launch_dbm)
    atten = build_attenuation(cfg)
    disp = at.Dispersion(o["dispersion_D_ps_nm_km"], o["dispersion_S_ps_nm2_km"],
                         1550.0, o["n_group"],
                         beta2_s2_m=o.get("beta2_s2_m"),
                         beta3_s3_m=o.get("beta3_s3_m"))
    eta = float(lk["eta_insertion_per_pass"])
    L_km = lk["length_km_per_traversal"] if length_km is None else length_km
    sc = ol.Scenario(
        dep=dep, atten=atten, disp=disp, length_m=float(L_km) * 1e3,
        eta_tx=np.sqrt(eta), eta_rx=np.sqrt(eta),
        mode_fwhm_hz=lk["mode_fwhm_ghz"] * 1e9,
        filter_fwhm_hz=lk["filter_fwhm_ghz"] * 1e9,
        filter_order=int(lk["filter_order"]),
        filter_isolation_db=lk["filter_isolation_db"],
        modeled_domain_nm=tuple(lk["modeled_domain_nm"]),
        e0_per_pass=lk["e0_per_pass"],
        raman_model=o["raman_model"], raman_scale=o["raman_scale"],
        temperature_k=o["temperature_k"],
        gamma_per_w_km=o["gamma_per_w_km"],
        fwm_convention=o.get("fwm_convention", "conventional_gamma"),
        fwm_profile_rms_hz=float(o.get("fwm_profile_rms_ghz", 0.0)) * 1e9,
        receiver_mode_convention=lk.get("receiver_mode_convention",
                                        "fixed_mode_overlap"),
        mod_rms_hz=o["mod_rms_ghz"] * 1e9,
        tx_broadband_floor_per_hz=o["tx_broadband_floor_per_hz"],
        leak_isolation_remote_db=o["leak_isolation_remote_db"],
        leak_isolation_local_db=o["leak_isolation_local_db"],
        xpm_kappa=o["xpm_kappa"], xpm_rel_fluct=o["xpm_rel_fluct"],
        xpm_band_hz=o["xpm_band_ghz"] * 1e9,
        gh_order=int(num["gauss_hermite_order"]),
        mode_line_window_sigma=float(num["mode_line_window_sigma"]))
    for k, v in (overrides or {}).items():
        setattr(sc, k, v)
    sc._fwm_cache = {}
    return sc


def effective_guard_hz(cfg):
    """Guard actually imposed on candidates: the nominal carrier guard plus the
    declared additional tuning margin.  Candidate boundary points are generated
    at g + m, never at g alone, so a zero-width component at exactly g does not
    survive a nonzero margin."""
    s = cfg["search"]
    return (float(s["guard_band_ghz"])
            + float(s.get("tuning_margin_ghz", 0.0))) * 1e9


def build_grid(cfg, dep, step_ghz=None):
    """Candidate frequency grid: uniform spacing plus every exact feature point.

    The guard boundaries, the carrier frequencies, the reserved-window edges and
    the hardware-domain edges are inserted explicitly, so an allowed component
    of zero width is still represented by a grid point and the feasible set does
    not depend on where the uniform grid happens to fall.
    """
    s = cfg["search"]
    lo_nm, hi_nm = s["hardware_domain_nm"]
    f_lo, f_hi = nm_to_hz(hi_nm), nm_to_hz(lo_nm)
    step = (s["grid_step_ghz"] if step_ghz is None else step_ghz) * 1e9
    guard = effective_guard_hz(cfg)
    extra = list(dp.guard_boundaries(dep, guard))
    extra += list(dep.freqs_hz)
    if s.get("reserved_window_nm"):
        rlo, rhi = s["reserved_window_nm"]
        extra += [nm_to_hz(rlo), nm_to_hz(rhi)]
    extra += [f_lo, f_hi]
    extra = [e for e in extra if f_lo <= e <= f_hi]
    return dp.candidate_grid(f_lo, f_hi, step, extra)


def build_masks(cfg, f_grid, dep):
    s = cfg["search"]
    lo_nm, hi_nm = s["hardware_domain_nm"]
    rw = s.get("reserved_window_nm")
    reserved_hz = (nm_to_hz(rw[1]), nm_to_hz(rw[0])) if rw else None
    return dp.domain_masks(f_grid, dep, effective_guard_hz(cfg),
                           nm_to_hz(hi_nm), nm_to_hz(lo_nm),
                           reserved_hz,
                           bool(s.get("restrict_quantum_to_reserved_window", False)))
