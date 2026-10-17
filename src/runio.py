"""Run directories, manifests, provenance, and CSV/JSON writing."""
import csv, hashlib, json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    src = os.path.join(ROOT, "src")
    return {f: file_hash(os.path.join(src, f))
            for f in sorted(os.listdir(src)) if f.endswith(".py")}


def package_versions():
    import numpy, scipy, matplotlib, pandas, yaml
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "pandas": pandas.__version__, "pyyaml": yaml.__version__,
            "platform": platform.platform()}


def run_dir(run_id):
    d = os.path.join(RESULTS, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        with open(path, "w") as fh:
            fh.write("")
        return path
    if fieldnames is None:
        # union of the keys in first-seen order, so rows that carry extra
        # columns (e.g. link-budget terms) do not lose them
        fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fieldnames})
    return path


def _fmt(v):
    if isinstance(v, (np.floating, float)):
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def columns_to_rows(cols, order=None):
    keys = order or list(cols.keys())
    n = len(cols[keys[0]])
    return [{k: cols[k][i] for k in keys} for i in range(n)]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_jdefault)
    return path


def _jdefault(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Manifest:
    def __init__(self, run_id, stage, config):
        self.d = {"run_id": run_id, "stage": stage, "started_utc": utcnow(),
                  "config": config, "software": package_versions(),
                  "source_sha256": source_hashes(),
                  "inputs": [], "outputs": [], "tolerances": {}, "notes": []}
        self._t0 = time.time()

    def add_input(self, path, category, note=""):
        rec = {"path": os.path.relpath(path, ROOT), "provenance_category": category,
               "note": note}
        if os.path.exists(path):
            rec["sha256"] = file_hash(path)
            rec["bytes"] = os.path.getsize(path)
        else:
            rec["sha256"] = None
            rec["status"] = "NOT SUPPLIED TO THIS RUN"
        self.d["inputs"].append(rec)

    def add_output(self, path):
        self.d["outputs"].append(os.path.relpath(path, ROOT))
        return path

    def note(self, text):
        self.d["notes"].append(text)

    def tolerance(self, k, v):
        self.d["tolerances"][k] = v

    def save(self, path):
        self.d["wall_time_s"] = round(time.time() - self._t0, 3)
        self.d["finished_utc"] = utcnow()
        return write_json(path, self.d)


# ---------------------------------------------------------------- provenance
PROVENANCE_CATEGORIES = ("measured", "manufacturer_specification",
                         "literature_model", "illustrative_assumption", "derived")


def provenance_rows(cfg=None, scenario=None):
    """Row-level provenance generated FROM THE EXECUTED CONFIGURATION.

    Every `value` below is read out of the effective configuration dictionary
    that produced the run (and, where given, the built `scenario` object), so a
    table row cannot drift away from the number the simulation used.  The
    provenance category, the reference plane and the sensitivity range are
    separate columns: an actual input value, where it comes from, and what was
    varied around it are three different statements.

    No measured optical data were supplied for this study: no measured Raman
    spectrum, no measured attenuation curve, no receiver characterization and
    no deployment record.  No row is therefore labelled `measured`.
    """
    if cfg is None:
        import config as cf
        cfg = cf.load("illustrative_baseline.yaml")
    P = "illustrative_assumption"
    L = "literature_model"
    M = "manufacturer_specification"
    D = "derived"
    dep = cfg["deployment"]; lk = cfg["link"]; o = cfg["optics"]
    se = cfg["search"]; tg = cfg["targets"]; num = cfg["numerics"]
    f_list = list(dep["classical_frequencies_ghz"])
    guard = float(se["guard_band_ghz"])
    margin = float(se.get("tuning_margin_ghz", 0.0))
    eta = float(lk["eta_insertion_per_pass"])

    def row(dataset, symbol, value, unit, plane, source, category, sens="-", note=""):
        return dict(dataset=dataset, symbol=symbol, value=value, unit=unit,
                    reference_plane_or_traversal=plane, source=source,
                    provenance_category=category, sensitivity_range=sens,
                    note=note)

    rows = [
        row("classical_allocation", "f_ci",
            f"{len(f_list)} frequencies, {min(f_list):.0f}-{max(f_list):.0f} GHz "
            f"on a 100 GHz grid; full list in classical_channels.csv",
            "GHz", "fibre input at terminal A",
            "configs/" + os.path.basename(cfg.get("_path", "")) +
            " deployment.classical_frequencies_ghz", P, "fixed in this study",
            "frozen allocation; the study optimizes the QUANTUM wavelength only"),
        row("classical_launch_power", "P_i0",
            f"{float(dep['classical_launch_dbm_per_channel']):.1f}", "dBm/channel",
            "classical transmitter output", "configuration value", P,
            (f"{cfg['map']['launch_dbm_min']:g} to {cfg['map']['launch_dbm_max']:g} "
             "dBm/channel in the operating map; link-budget minimum per receiver"
             if "map" in cfg else "-15 to 0 dBm/channel in the operating map"),
            "identical for all carriers"),
        row("classical_direction", "-", str(dep["classical_direction"]), "-",
            "span", "configuration value", P, "-",
            f"quantum traversal directions {list(dep['quantum_directions'])}"),
        row("length_per_traversal", "L", f"{float(lk['length_km_per_traversal']):.3f}",
            "km", "per traversal", "configuration value", P,
            "1-30 km in the operating map", "three traversals of equal length"),
        row("insertion_transmission", "eta_tx, eta_rx", f"sqrt({eta:g}) each",
            "field amplitude", "launch and receive planes of each traversal",
            "configuration link.eta_insertion_per_pass", P, "-",
            "the per-pass 0.82 is split symmetrically; component contents were "
            "never specified"),
        row("accepted_mode", "psi(f)",
            f"Gaussian spectral intensity, {float(lk['mode_fwhm_ghz']):.1f} GHz FWHM",
            "1/Hz", "receiver mode at every traversal output",
            "configuration link.mode_fwhm_ghz", P, "-",
            "declared mode, not an inferred hardware property"),
        row("receiver_filter", "T_q(f)",
            f"super-Gaussian order {int(lk['filter_order'])}, "
            f"{float(lk['filter_fwhm_ghz']):.1f} GHz FWHM, "
            f"{float(lk['filter_isolation_db']):.0f} dB floor, modeled domain "
            f"{lk['modeled_domain_nm'][0]:.0f}-{lk['modeled_domain_nm'][1]:.0f} nm",
            "power response", "quantum receiver input",
            "configuration link.filter_*", P, "-",
            "finite modeled domain keeps B_eq finite"),
        row("receiver_mode_convention", "eta_mode",
            str(lk.get("receiver_mode_convention", "fixed_mode_overlap")),
            "-", "each traversal output",
            "configuration link.receiver_mode_convention", P,
            "power_average compared as a scenario",
            "fixed_mode_overlap = |int |psi|^2 sqrt(T_q) df|^2; the signal "
            "projection and the noise projection use the SAME fixed mode"),
        row("fibre_attenuation", "alpha(lambda)",
            (f"tabulated {o.get('attenuation_csv')}"
             if o.get("attenuation_source", "tabulated_smf28") == "tabulated_smf28"
             else f"analytic three-term anchored to "
                  f"{float(o['attenuation_anchor_db_km']):.3f} dB/km at "
                  f"{float(o['attenuation_anchor_nm']):.0f} nm"),
            "dB/km", "span",
            ("data/raw/attenuation_smf28.csv, per-row provenance in Table S1b"
             if o.get("attenuation_source", "tabulated_smf28") == "tabulated_smf28"
             else "standard Rayleigh + IR + constant form"),
            M if o.get("attenuation_source", "tabulated_smf28") == "tabulated_smf28" else P,
            "analytic three-term model compared as a scenario",
            "datasheet-style rows are manufacturer maxima at listed wavelengths; "
            "interpolated values are NOT measurements and NOT bounds"),
        row("dispersion", "D, S, n_g",
            f"D={float(o['dispersion_D_ps_nm_km']):g} ps/(nm km), "
            f"S={float(o['dispersion_S_ps_nm2_km']):g} ps/(nm^2 km), "
            f"beta2={float(o['beta2_s2_m']):.4g} s^2/m, n_g={float(o['n_group']):g}",
            "mixed", "span", "typical single-mode fibre values", P, "-",
            "used for the FWM phase mismatch and the XPM walk-off"),
        row("nonlinear_coefficient", "gamma", f"{float(o['gamma_per_w_km']):g}",
            "1/(W km)", "span", "typical single-mode fibre value", P, "-",
            "conventional definition gamma = 2 pi n2 /(lambda A_eff)"),
        row("fwm_convention", "d",
            str(o.get("fwm_convention", "conventional_gamma")), "-", "span",
            "declared numerical convention", D, "-",
            "pumps enumerated with i<=j; amplitude degeneracy 1 (i==j) and "
            "2 (i!=j) with the conventional gamma; k != i,j"),
        row("fwm_spectral_profile", "g_l(f)",
            f"{float(o.get('fwm_profile_rms_ghz', 0.0)):g} GHz rms "
            f"({'narrow-line (delta) limit' if float(o.get('fwm_profile_rms_ghz', 0.0)) == 0 else 'prescribed Gaussian profile'})",
            "GHz", "generated component", "declared numerical convention", P,
            "0, 1, 5, 25 GHz rms compared as a diagnostic",
            "a prescribed broadening at fixed product power; NOT a simulation of "
            "modulated four-wave mixing"),
        row("raman_calibration_point", "rho_meas", "2.85e-9", "1/(km nm)",
            "span", "Qi, Zhu, Qian, Lo, New J. Phys. 12, 103042 (2010)", L, "-",
            "ONE point at pump 1559.79 nm / observation 1554.94 nm "
            "(delta_f = -0.5995 THz, anti-Stokes).  Polarization convention "
            "resolved from the source: Eq. (5) -> (6) -> (15) apply the factor "
            "1/2 for single-polarization selection only at the last step, so "
            "this coefficient is polarization-summed"),
        (row("raman_lineshape", "g(delta_f)", str(o["raman_model"]),
             "-", "span",
             "Hollenbeck and Cantrell, J. Opt. Soc. Am. B 19, 2886 (2002), Table 1",
             L, "two stand-in analytic shapes; level x0.5, x2 and the da Silva "
             "et al. (2014) level",
             "13-mode intermediate-broadening fit to the measured silica gain; "
             "Bose factor n_B+1 (Stokes) and n_B (anti-Stokes) applied per side; "
             "level fixed at the calibration point")
         if o["raman_model"] == "hollenbeck_cantrell_2002" else
         row("raman_lineshape", "g(delta_f)", str(o["raman_model"]),
             "-", "span", "stand-in analytic shape defined in src/raman.py", P,
             "level x0.5 and x2",
             "NOT a fit to a published measured spectrum; one calibration point "
             "does not supply a broadband spectrum")),
        row("raman_level_scale", "scale", f"{float(o['raman_scale']):g}", "-",
            "span", "configuration value", P, "0.5 - 2.0",
            "stress factor, not a measured uncertainty; the near-carrier "
            "measurement of da Silva et al., J. Lightwave Technol. 32, 2332 "
            "(2014) is carried as a separate level scenario"),
        row("raman_thermal_factor", "n_B + 1", f"{float(o['temperature_k']):g} K",
            "-", "span", "Bose-Einstein detailed balance", L, "-", ""),
        row("leakage_path_isolation", "iso_remote, iso_local",
            f"{float(o['leak_isolation_remote_db']):.0f} dB remote, "
            f"{float(o['leak_isolation_local_db']):.0f} dB local", "dB",
            "classical launch plane to quantum FILTER INPUT",
            "configuration values", P, "-10 dB stress",
            "declared to EXCLUDE the final quantum filter so it is not double "
            "counted"),
        row("transmitter_broadband_floor", "s_floor",
            f"{float(o['tx_broadband_floor_per_hz']):g}", "1/Hz of carrier power",
            "classical transmitter output", "declared assumption", P, "x10 stress",
            "dominates in-band leakage: the receive filter is centred on the "
            "quantum mode and cannot reject an in-band transmitter floor"),
        row("classical_modulation_rms", "B_mod", f"{float(o['mod_rms_ghz']):g}",
            "GHz", "classical transmitter output", "configuration value", P, "-",
            "sets the modulated carrier width used by the leakage model"),
        row("xpm_differential_coupling", "kappa", f"{float(o['xpm_kappa']):g}", "-",
            "per traversal", "declared assumption", P, "0.25 x stress",
            "differential H-V coupling fraction; identical coupling gives zero "
            "differential phase"),
        row("xpm_power_fluctuations", "S_dP(F)",
            f"flat two-sided PSD, {100*float(o['xpm_rel_fluct']):.0f}% rms depth, "
            f"{float(o['xpm_band_ghz']):g} GHz band", "W^2/Hz",
            "per traversal", "declared assumption", P, "-",
            "Gaussian phase statistics are ASSUMED; an on-off keyed intensity is "
            "not Gaussian merely because a variance was computed"),
        row("base_error", "e_0", f"{float(lk['e0_per_pass']):g}",
            "probability per pass", "per traversal", "declared assumption", P,
            ", ".join(f"{100*float(a['e0']):g}%" for a in
                      cfg.get("base_error_sweep", {}).get("anchors", []))
            or "x0.5 - x2",
            "basis error of state preparation, polarization drift and analysis; "
            "swept over published anchors (Agnesi 2020, Bolanos 2026, Xavier "
            "2009) and the largest admissible value is reported; independent of "
            "the additive background by assumption"),
        *[row(f"classical_receiver_{rx['id']}", "S_rx",
              f"{float(rx['rx_sensitivity_dbm']):g}", "dBm",
              "classical receiver input", str(rx["source"]),
              rx.get("provenance_category",
                     L if "measured" in str(rx["source"]) else M), "-",
              f"{rx['label']}; sets the link-budget minimum launch power")
          for rx in cfg.get("classical_link", {}).get("receivers", [])],
        *([row("classical_link_terms", "IL_mux, IL_demux, M",
               f"{float(cfg['classical_link']['mux_insertion_db']):g}, "
               f"{float(cfg['classical_link']['demux_insertion_db']):g}, "
               f"{float(cfg['classical_link']['system_margin_db']):g}", "dB",
               "classical link", "declared assumption", P, "-",
               "P_min = S_rx + alpha_c L + IL_mux + IL_demux + M with alpha_c the "
               "worst classical attenuation")]
          if "classical_link" in cfg else []),
        row("guard_band", "g", f"{guard:g}", "GHz", "candidate domain",
            "configuration search.guard_band_ghz", P, "49, 50, 51 GHz",
            "with 100 GHz carrier spacing the between-carrier allowed width is "
            "100 - 2g"),
        row("tuning_margin", "m", f"{margin:g}", "GHz", "candidate domain",
            "configuration search.tuning_margin_ghz", P, "0, 1, 2, 2.5, 5 GHz",
            "ILLUSTRATIVE scenario choice, not a measured hardware tolerance; "
            "candidates are generated at the effective guard g+m"),
        row("search_domain", "Omega_q",
            f"{se['hardware_domain_nm'][0]:.0f}-{se['hardware_domain_nm'][1]:.0f} nm, "
            f"{float(se['grid_step_ghz']):g} GHz grid with exact boundary points",
            "nm / GHz", "candidate domain", "configuration search.*", P,
            f"refined to {float(se['refine_step_ghz']):g} GHz",
            "zero-width allowed components are isolated points, not intervals"),
        row("targets", "F_min, Y_min, epsilon",
            f"{float(tg['F_min']):g}, {float(tg['Y_min']):g}, {float(tg['epsilon']):g}",
            "-", "logical block", "declared illustrative targets", P,
            f"F_min {tg['F_min_sensitivity']}, Y_min {tg['Y_min_sensitivity']}",
            "application thresholds, NOT security thresholds"),
        row("fixed_reference_wavelength", "lambda_ref",
            f"{float(tg['fixed_reference_nm']):g}", "nm", "candidate domain",
            "declared before any optimum was examined", P, "-",
            "comparator rule; reported as invalid when it is not an allowed "
            "candidate"),
        row("numerics", "GH order, window",
            f"Gauss-Hermite order {int(num['gauss_hermite_order'])} "
            f"(check {int(num['gauss_hermite_check_order'])}), line window "
            f"{float(num['mode_line_window_sigma']):g} sigma", "-", "quadrature",
            "configuration numerics.*", D, "-", ""),
        row("code_and_recovery", "[[7,1,3]]",
            "Steane CSS code, fixed Hamming syndrome recovery", "-",
            "logical block", "Steane 1996; explicit construction verified in code",
            L, "-", ""),
        row("composition_and_bound", "T, nu, c0, c1, s, r, q, C, Y",
            "src/optical_channel.py, src/local_protocol.py, src/reliability.py",
            "-", "-", "manuscript equations", D, "-", "derived quantities"),
    ]
    if scenario is not None:
        rows.append(row("effective_scenario_check", "eta_tx*eta_rx, L",
                        f"{float(scenario.eta_tx * scenario.eta_rx):.6f}, "
                        f"{scenario.length_m / 1e3:.3f} km", "-", "per traversal",
                        "built Scenario object", D, "-",
                        "read back from the object the run actually used"))
    return rows
