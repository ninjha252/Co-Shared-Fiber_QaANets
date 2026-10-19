# Validation report

What the numerics verify, and what the physical model merely assumes.  The
optical scenario is **illustrative and uncalibrated**: no measured fibre
spectrum, receiver characterization or deployment record was supplied.

## 1. Numerical verification (independent of any fibre measurement)

91 asserted checks, 0 failures, 15 report-only diagnostics.
Sweep sizes are separate: 68 optical dilation points, 272
decoder sweep rows, 306 key-sensitivity rows.

* **Accepted optical channel.**  An independent thermal-environment
  beam-splitter dilation, exact within each total-photon-number sector,
  reproduces `c0 = T/(1+nu)^4` and `c1 = 2 nu (1+nu-T)/(1+nu)^4` over
  68 (T, nu) points: maximum accepted-Choi trace-norm error
  5.7e-16 (target 1e-9) and maximum normalized-channel error
  `|| J_num/s_num - J_ref/s_ref ||_1` = 1.1e-15 (target 1e-7).
  Acceptance is input independent to 0e+00 over six polarization probe
  states; positivity, trace nonincrease and `Tr_out J = s I/2` hold.
* **Composition.**  Three explicit Fock-space traversals with final-only
  acceptance reproduce the single effective channel to 2e-16,
  confirming `nu = eta_3 eta_2 nu_1 + eta_3 nu_2 + nu_3`.
* **Logical recovery.**  Exact enumeration of all 4^7 net Pauli strings and an
  independent 8-qubit density-matrix recovery agree to 1e-13.  Across the
  sweep `F_e >= C_7,1(q)` holds, maximum gap 0.438, most negative
  residual -1e-15.  The synthetic benchmark is reproduced to
  3e-13 relative.
* **Four-wave mixing.**  Independent low-conversion benchmarks that do not call
  the production prefactor confirm the corrected normalization; the earlier
  convention is rejected at nine times the benchmark power.

This verifies the mathematics of the reduction and the recovery.  It does not
show that a physical fibre background is thermal, independent, or of the assumed
spectral shape.

## 2. Physical assumptions (conditional; not verified here)

1. The Raman spectrum is the Hollenbeck-Cantrell (2002) intermediate-broadening
   fit to the measured gain of fused silica, with the Bose factor applied on each
   side and the level fixed at the measured coefficient of Qi et al. (2010,
   -0.5995 THz anti-Stokes).  The polarization convention is resolved from that
   source (Eqs. 5 -> 6 -> 15): the coefficient is polarization summed and the
   factor 1/2 is applied once, in the occupancy integral.  An independent
   near-carrier measurement (da Silva et al. 2014) lies a factor of about 4
   below this level at 0.6 THz and grows linearly with offset where the model is
   nearly flat; it is carried as the scenario `raman_level_dasilva2014`.
2. The aggregate background is treated as an independent isotropic thermal
   effective state.  Adding Raman, FWM and leakage powers does not establish
   thermal photon statistics, nor independence across qubits and traversals.
3. The receiver is a fixed-mode projection with the declared mode restoration at
   every traversal output.  The full-chain spectral reference differs by up to
   0.63% in `T` and 4.5% in `Y_lb`: a model choice, not a
   numerical error.
4. FWM components are delta lines in the main results; prescribed profiles of
   1, 5 and 25 GHz rms are phenomenological, not a modulated-FWM simulation.
5. The XPM model is a flat two-sided power-fluctuation PSD of
   100% rms depth over
   25 GHz with differential coupling
   kappa = 1, converted to a Gaussian phase.
   Gaussian phase statistics are assumed, not derived from an on-off keyed
   intensity, and phase uniformity over the accepted pulse is assumed.
6. Backward FWM generation, reflection and Rayleigh backscatter are not modeled,
   so the return traversal receives no modeled FWM.
7. The 1 GHz tuning margin is an illustrative scenario choice, not a
   measured hardware tolerance; 2.5 GHz is included as a sensitivity case,
   motivated by the EXFO FTBx-2850 frequency uncertainty specification.
8. Only one datasheet anchor (1550 nm) lies inside the quantum band; the rest of
   the attenuation curve there is interpolated, and a manufacturer maximum is
   neither a measurement nor a bound at interpolated wavelengths.
9. The classical launch power at the link-budget minimum uses the literature
   sensitivity at 1.25 Gb/s and the assumed -24 dBm benchmark at 10.7 Gb/s,
   the worst classical attenuation, and assumed 3 dB output demultiplexer
   and 3 dB system margin terms. Powers are at the fiber input; the 3 dB
   input multiplexer loss applies only to the upstream transmitter power.

## 3. Sensitivity

Of the enumerated scenarios at 5 km / -10 dBm/ch, 2 empty the
quality-certified domain: `raman_scale_2.0` (M = 0.9861), `base_error_doubled` (M = 0.9861).  Classification uses the full-domain maximum
`M_theta = max_(s>0) C_7,1(q)`; `M_theta < F_min` means no candidate is certified
by this bound, not that exact fidelity fails.  The nominal wavelength held
fixed is certified under 13 of 16 scenarios; it also fails under `raman_shape_standin_multigaussian`, where retuning still finds a certified candidate.
The intersection of the certified domains over all scenarios is
empty.

Prescribed FWM broadening lowers the peak occupancy (x0.168 at
25 GHz rms) but raises the occupancy at the selected wavelength from
3.2e-44 to 1.0e-05 (0.34% of nu there),
because that candidate is fed only by profile tails.  The certified set changes
by 0 candidates and `Y_lb` by +0.0028%, so the narrow-line
approximation does not drive the conclusions at these parameters; the tails are
qualitatively important and quantitatively small here.

Tuning margin, at the four predeclared points:

| Operating point | lambda* at 0 GHz margin | lambda* at 1 GHz margin | displacement (nm) | relative change in Y_lb |
|---|---|---|---|---|
| 2 km, -15 dBm/ch | 1550.5170 | 1551.3273 | +0.8104 | -0.160% |
| 3 km, -13 dBm/ch | 1550.5170 | 1551.3273 | +0.8104 | -0.239% |
| 5 km, -10 dBm/ch | 1550.5170 | 1551.3273 | +0.8104 | -0.373% |
| 10 km, -5 dBm/ch | undefined | undefined | undefined | undefined |

## 4. Convergence

Grid refinement (2 -> 1 -> 0.5 GHz at fixed quadrature) and quadrature
refinement (Gauss-Hermite 48 -> 96 at fixed grid) were varied separately at all
four predeclared points.  Worst relative change in the optimized `Y_lb`
1.1e-09 (target 1e-3); worst absolute change in `C` at the reported
selections 2.8e-10 (target 1e-6).  The certification boundary is
resolved to 9.4e-07 in `C`.

## 5. Unresolved

* No Raman spectrum, receiver characterization or attenuation measurement was
  taken for this fibre; the scans use published models and datasheet values, so
  every physical scan is conditional on them.
* A physically derived FWM profile needs the modulation and coherence model of
  the classical carriers; prescribed widths only bound the direction of the
  effect.
* Independence of noise realizations across qubits and traversals is assumed;
  the correlated-block control is a diagnostic, not a proof.
* The sampled key range is not a worst-case guarantee over private keys and does
  not measure Pauli-twirl error.

## 6. Manuscript operating-point comparisons

| Operating point | Role | allowed | Omega_F | lambda* (nm) | C | F_e (identity) | Y_lb | status |
|---|---|---|---|---|---|---|---|---|
| 2 km, -15 dBm/ch | interior example | 454 | 454 | 1551.3273 | 0.997996 | 0.998447 | 2.1912e-03 | certified for F_min and Y_min |
| 3 km, -13 dBm/ch | interior example | 454 | 454 | 1551.3273 | 0.997297 | 0.997906 | 9.1569e-04 | certified for F_min and Y_min |
| 5 km, -10 dBm/ch | original comparison point | 454 | 454 | 1551.3273 | 0.992384 | 0.994111 | 1.6321e-04 | certified for F_min and Y_min |
| 6 km, -10 dBm/ch | active fidelity constraint comparison | 454 | 51 | 1537.7836 | 0.991331 | 0.993298 | 6.0774e-05 | certified for F_min and Y_min |
| 10 km, -5 dBm/ch | legacy comparison point | 454 | 0 | undefined | max C = 0.870773 | undefined | undefined | NOT certified: Omega_F empty |
