# Quantum–classical coexistence in a single fibre: simulation code

Simulation and analysis code for the manuscript on quantum–classical
coexistence over a single fibre. It evaluates an optical-to-qubit reduction of
the physical channel, a correctable-history fidelity bound for a
[[7,1,3]] Steane block, and the quantum-wavelength selection that follows from
them under a fixed classical allocation.

The repository reproduces every figure and table in the paper from the stored
configurations, and it re-derives the stored results from scratch with one
command.

## Two separate questions

The code keeps these apart, and so does the output:

1. **Numerical verification.** Do independent constructions reproduce the
   derived optical channel and satisfy the recovery bound? This is a statement
   about the mathematics and needs no fibre measurement.
2. **Model application.** Which wavelength and which operating region does the
   specified optical scenario certify? This is conditional on the physical
   inputs, the receiver model and the noise assumptions.

**No optical measurement was taken for this study.** The Raman spectrum is the
published Hollenbeck–Cantrell fit to the measured gain of fused silica,
normalized at the measured coefficient of Qi et al. (2010); the attenuation is
tabulated from manufacturer maxima; the classical receivers are a measured
sensitivity (Patel et al. 2012) and a datasheet value; the base error is swept
over published polarization-QKD values. Every input is recorded per row with
its source in `results/*/provenance.csv` and in Main Table 1, and every
wavelength reported here is conditional on those inputs.

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Tested with Python 3.12.14, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5,
matplotlib 3.11.1, PyYAML 6.0.3 and pytest on macOS (arm64). No compiled
extensions and no network access are required.

## Reproducing the study

```bash
.venv/bin/python scripts/run_all.py quick   # validation + unit tests, ~40 s
.venv/bin/python scripts/run_all.py full    # everything, ~45 min
```

`full` runs, in order:

| Stage | Script | What it produces |
|---|---|---|
| Logical validation | `validate_decoder.py` | encoder, recovery and bound checks; the noise sweep |
| Optical validation | `validate_optics.py` | impairment-model checks; the independent optical construction |
| Unit tests | `pytest` | fast regression tests, including fault injection |
| Operating points | `run_baseline.py` × 8 | the four predeclared points with a 1 GHz tuning margin, and the same four with none |
| Reserved window | `run_baseline.py` | the restricted-search comparison |
| Operating map | `run_operating_map.py` | 30 lengths × 31 launch powers, with the largest admissible base error at each point |
| Guard and margin | `run_guard_margin.py` | guards 49/50/51 GHz × margins 0/1/2/2.5/5 GHz |
| Boundary | `run_boundary_refine.py` | `max_λ C = F_min` by bracketed bisection |
| FWM profile | `run_profile_width.py` | delta lines versus 1, 5 and 25 GHz rms profiles |
| Link budget | `run_link_budget.py` | operating points at the classical link-budget minimum; base-error sweep over published values |
| Sensitivity | `run_sensitivity.py` | ablations, scenarios, robust selection |
| Outputs | `make_figures.py`, `make_tables.py`, `make_reports.py` | figures, tables and reports, from stored CSVs only |

Figures and tables never recompute physics: they read the CSVs written by the
run scripts, so a plot cannot disagree with the data behind it.

### Running one operating point

`scripts/run_baseline.py` accepts environment overrides, which are written back
into the configuration before anything is computed:

```bash
QCFC_RUN_ID=op_5km_m10dBm QCFC_LENGTH_KM=5 QCFC_LAUNCH_DBM=-10 \
QCFC_MARGIN_GHZ=1.0 .venv/bin/python scripts/run_baseline.py
```

Also accepted: `QCFC_BASELINE_CONFIG` (which configuration file to load) and
`QCFC_FWM_PROFILE_GHZ` (prescribed four-wave-mixing profile width).

## Layout

```
configs/     illustrative_baseline.yaml (main scenario), validation.yaml
             (synthetic verification), legacy_reserved_window.yaml
             (restricted-search comparison)
data/raw/    attenuation_smf28.csv, datasheet maxima with per-row provenance
src/         the model: units, deployment, modes, raman, fwm, leakage, xpm,
             attenuation, optical_channel, optical_dilation, optical_link,
             mode_reference, local_protocol, steane, protocol, reliability,
             optimization, legacy_compare, runio
scripts/     validation, scans, sensitivity studies, figures, tables, reports
tests/       fast regression tests, including the fault-injection checks
results/     stored outputs of the recorded runs (see below)
```

### What is in `src/`

| Module | Contents |
|---|---|
| `units.py` | SI constants and the unit conversions used at the input and output boundaries |
| `deployment.py` | the fixed classical allocation, propagation directions and reference planes |
| `attenuation.py` | tabulated and analytic attenuation, dispersion, per-row provenance |
| `modes.py` | the accepted mode, the receiver filter, and mode occupancy, including profiled lines |
| `raman.py` | spontaneous Raman background: the Hollenbeck–Cantrell silica gain model, calibration at the measured detuning, the da Silva et al. near-carrier measurement, and two stand-in lineshapes kept for sensitivity |
| `fwm.py` | four-wave-mixing products, the degeneracy convention and the longitudinal integral |
| `leakage.py` | classical filter leakage on the local and remote paths |
| `xpm.py` | residual differential cross-phase modulation, closed form and reference routes |
| `optical_link.py` | assembles per-traversal (η, ν, σ²_φ) for a candidate wavelength |
| `optical_channel.py` | the accepted optical operation: c₀, c₁, s, r and the three-traversal composition |
| `optical_dilation.py` | the independent beam-splitter dilation used to check that operation |
| `mode_reference.py` | the full-chain spectral reference for T and ν |
| `local_protocol.py` | private operations, traversal channels, and the affected-location probability |
| `steane.py` | the [[7,1,3]] code, the fixed CSS recovery and two exact fidelity routes |
| `protocol.py` | reduction of a parameter set to every scalar of the certificate |
| `reliability.py` | correctable weight, acceptance and yield bounds |
| `link_budget.py` | minimum classical launch power from receiver sensitivity, span loss, multiplexing optics and margin |
| `optimization.py` | the wavelength scan, feasible sets and selection rules |
| `legacy_compare.py` | the quantitative record of corrections to the earlier implementation |
| `runio.py` | run manifests, hashing, CSV and JSON writers, and the provenance table |

## Outputs

`results/` holds the outputs of the recorded runs:

- `figures/`: the four main figures and Supplementary Figures S1 and S2, as
  PDF, SVG and PNG. `figures/final/` (and the identical `figures/nocaption/`)
  holds the manuscript versions: no figure title on the canvas, Times New
  Roman, larger axis fonts.
- `tables/`: Main Tables 1 and 2 and Supplementary Tables S1–S3, each as
  CSV, Markdown and LaTeX; `manuscript_parameter_table.tex` is the compact
  parameter table for the paper; `tables/supporting/` holds the full four-wave-mixing
  accounting, the receiver-model comparison, the correlated-fault control, the
  selector comparison and the certified regions.
- `validation/`, `op_*/`, `illustrative_baseline/`, `legacy_reserved_window/`:
  the per-run CSVs and the `run_manifest_*.json` of each stage.
- `validation_report.md`, `captions.md`, `results_evidence_outline.md`,
  `run_metadata.json`: the written record, every number read from a CSV.
- `CHANGES.md` at the repository root records each correction made to the
  earlier implementation, its mathematical basis, and what it changed.

Per-candidate scans (`wavelength_scan.csv`) are stored gzip-compressed to keep
the repository small; the readers accept either form, and rerunning a stage
writes the uncompressed file.

Every run script writes a manifest containing the resolved configuration, input
file SHA-256 hashes, seeds, package versions, source file hashes, tolerances,
wall time and output paths. `results/run_metadata.json` aggregates them.

## Conventions worth knowing before reading the code

- Internal units are SI: metres, seconds, hertz, watts, radians. Conversion to
  km, GHz, nm and dBm happens only at the input and output boundaries.
- `a` is the **power** attenuation in m⁻¹; a field amplitude decays as
  `exp(-a z / 2)`.
- `delta_f = f_classical - f_observed`: positive is Stokes, negative is
  anti-Stokes.
- Occupancy ν is **per polarization**. The reported Raman coefficient is
  polarization summed, and the factor 1/2 is applied once, in the occupancy
  integral.
- Four-wave mixing uses the conventional γ with pump pairs enumerated as
  `i ≤ j`, `k ∉ {i, j}`, and amplitude multiplicity 1 (i = j) or 2 (i ≠ j).
  The alternative bookkeeping 3 and 6 requires γ/3; the two must not be mixed.
- The receiver uses one convention for signal and noise: both are projected
  onto the same fixed mode, `η = |∫ |ψ(f)|² t(f) df|²` with `t = √T_q`. The
  power average is retained only as a declared sensitivity scenario.
- The effective guard is the carrier guard plus the added tuning margin, and
  candidate boundary points are generated at the sum, so a zero-width component
  at exactly the guard does not survive a nonzero margin.

