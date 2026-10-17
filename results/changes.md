# Changes

Corrections applied to the earlier version of this code before any physical
scan was rerun, with the mathematical basis of each and the results that
changed.  Every number below is read from the regenerated outputs.

## 1. Four-wave-mixing normalization

**What was wrong.**  Pump pairs were enumerated with `i <= j` and given the
amplitude degeneracy `d = 3` (i == j) and `d = 6` (i != j) while the
*conventional* fibre coefficient `gamma = 2 pi n2 / (lambda A_eff)` was used
unchanged.

**Basis.**  Expanding `dA/dz = i gamma |A|^2 A` for a multi-frequency field and
counting each unordered pump pair once gives amplitude coefficients 1 and 2 with
the conventional gamma; the 3 / 6 bookkeeping belongs with `gamma/3`.  Using
both overcounts the generated amplitude by 3 and the power by 9.

**Fix.**  `src/fwm.py` takes an explicit `convention`: `conventional_gamma`
(`d = 1, 2`, the production setting) or `gamma_over_three` (`d = 3, 6` with
`gamma/3`).  The choice is recorded in the audit block and in Main Table 1.

**Independent tests.**  Lossless, phase-matched seeded benchmarks that do not
call the production prefactor: degenerate `P = gamma^2 P_p^2 P_s L^2` and
nondegenerate `P = 4 gamma^2 P_i P_j P_k L^2`, plus `L^2`, `gamma^2` and cubic
launch-power scalings, and a fault injection in which the earlier convention
gives exactly nine times the benchmark power and is rejected (2 benchmark
rows and 1 fault-injection row in Table S1; tests
`test_fwm_degenerate_benchmark_and_factor_of_nine_rejection`,
`test_fwm_nondegenerate_benchmark`).

**What changed.**  Every four-wave-mixing occupancy falls by a factor of 9 at
fixed launch power, which enlarges the certified sets and moves the
certification boundary.  The effect cannot be isolated from the present
results, because the receiver convention, the tuning margin and the candidate
grid were corrected in the same revision.

## 2. Parameter tables generated from the executed configuration

`src/runio.py:provenance_rows` no longer holds hard-coded physical values: it
reads the effective configuration of the run, so the exported table cannot
drift away from the numbers the simulation used.  The mismatches found in the
earlier table are gone by construction:

| Input | Earlier table | Executed configuration (now exported) |
|---|---|---|
| Attenuation | analytic, 0.190 dB/km anchor | tabulated SMF-28 table, 0.180 dB/km at 1550 nm |
| Filter order | 3 | 4 |
| XPM differential coupling | 0.05 | 1 |
| Relative power fluctuation | 10% rms | 100% rms |
| XPM fluctuation bandwidth | 10 GHz | 25 GHz |

Value, provenance category, reference plane and sensitivity range are separate
columns.  Interpolated datasheet rows are labelled as interpolations, not as
measurements and not as bounds.

## 3. Tolerances and validation counts

The validation helper accepted a generic dimensionful tolerance, so a
coefficient of order 1e-23 passed on the absolute criterion whatever its
relative error.  Tolerances are now relative wherever the reference is nonzero
and absolute only for zero limits and dimensionless residuals; a check with no
stated tolerance raises.  `test_raman_calibration_check_rejects_a_wrong_
normalization` shows a coefficient inflated by 1e6 failing the corrected
comparison and passing the earlier one.

Counts are reported separately: **91 asserted checks (0 failures)
and 15 report-only diagnostics**, with sweep sizes quoted per stage.  The
earlier count mixed these two categories.

## 4. Receiver model

The signal used a power average `int |psi|^2 T_q df` while the noise was
projected into a fixed Gaussian mode.  Both now use the same fixed-mode
projection `eta_mode = |int |psi|^2 t(f) df|^2` with `t = sqrt(T_q)`.  For the
10 GHz mode and the 25 GHz order-4 filter the two differ by 0.991466 versus
0.990210.  The power average is retained only as a declared scenario
(`receiver_power_average`) and is never mixed into a main result.

`src/mode_reference.py` adds an independent full-chain spectral reference
(`T_ref`, `nu_ref`) that keeps the frequency dependence through all three
traversals and projects once at the end.  Against the scalar recurrence it
differs by up to 0.63% in `T`, 0.36% in `nu`, 1.2e-03 in `C`
and 4.5% in `Y_lb`.  The main results keep the scalar recurrence, whose
physical content is that each terminal re-launches into the specified mode: the
per-traversal modal overlap is the declared cost of that restoration.  No free
mode restoration is introduced, and the size of the difference is reported.

## 5. FWM spectral profile

Generated components were delta lines.  `modes.occupancy_from_profiled_lines`
integrates a normalized profile of prescribed rms width against the receiver
mode, with the delta line as the exact zero-width limit (verified against
`occupancy_from_line` and against the analytic Gaussian convolution).  Widths of
0, 1, 5 and 25 GHz are compared at all four predeclared operating points.

## 6. Candidate domain, margin and selectors

* The effective guard is `g + m`, and candidate boundary points are generated at
  `g + m`: a zero-width component at exactly `g` does not survive a nonzero
  margin.  The main configuration uses `g = 50 GHz`, `m = 1 GHz`.
* The predeclared fixed reference is an EXACT rule: if that wavelength is not an
  allowed candidate the rule is reported invalid, and the nearest allowed
  candidate is a separate, separately labelled comparator.
* Exact identity-key fidelity is computed for every allowed candidate by Pauli
  enumeration, not at a handful of points.
* Grid refinement and quadrature refinement are varied separately.

## 7. Boundary solve

The certification boundary is a root of `max_lambda C - F_min`, found by
bracketed bisection on directly re-evaluated maxima.  Every sign change on the
sampled grid is solved separately (monotonicity is not assumed) and the subset
used for the inner wavelength maximization is verified against the full allowed
set at the root.  Worst residual over the converged powers 9.4e-07 in
`C` against a 1e-6 target; 0 unresolved; 0 powers with more
than one sign change.

## 8. What the corrections changed in the reported results

| Quantity | Earlier version | This version |
|---|---|---|
| FWM power normalization | nine times too large | benchmarked and corrected |
| Receiver signal transmission | power average 0.991466 | modal overlap 0.990210 |
| Candidate domain | guard 50 GHz, no margin | guard 50 GHz + 1 GHz margin |
| Exact fidelity curve | <= 80 sampled candidates | every allowed candidate |
| Fixed reference 1550 nm | snapped to the nearest allowed candidate | reported invalid, nearest allowed listed separately |
| Certification boundary | interpolated, verified | bracketed bisection, residual 9.4e-07 in C |
| Validation counting | mixed single count | 91 asserted + 15 diagnostics |

## 9. Submission reproducibility update

Finite-width FWM integration now uses the analytic product of the normalized
line profile and accepted Gaussian mode before quadrature. The physical
integrand, profile powers and delta-line branch are unchanged. Direct adaptive
quadrature regression checks include the filter and photon-energy factor.

Interval summaries on compressed allowed grids retain the original full-grid
indices. Excluded carrier guards therefore separate intervals even when every
allowed candidate is certified; this changes counts, not physical predictions.

The full workflow now exports the 6 km, -10 dBm/ch example in Main Table 2.
The original four diagnostic anchors and their margin comparisons are retained.
The -24 dBm receiver benchmark is labelled as an illustrative assumption,
matching the manuscript; its numerical value is unchanged.
