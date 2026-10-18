# Verification of the submission update

All 22 stages listed in the updated `scripts/run_all.py` completed using the
existing scripts. Validation and the first baseline ran sequentially; the
remaining independent numerical stages ran concurrently, followed by figures,
tables and reports. Boundary refinement was rerun after its operating-map
input became available. The 3 km baseline was rerun after an incomplete export
associated with the interrupted initial run. Every final scan was checked for
the full configured grid size and unique frequencies before packaging.

- 35 regression tests passed.
- 91 independent asserted validation checks passed; 15 additional rows are
  report-only diagnostics, not counted as passed tests.
- Ten baseline directories were regenerated: five main operating points,
  four zero-margin comparisons, and the legacy reserved-window comparison.
- All nine previously supplied baseline scans match their original numerical
  outputs within relative tolerance 1e-11 and absolute tolerance 1e-20.
  Their sampled classifications are unchanged.
- The 6 km, -10 dBm/ch result reproduces the existing manuscript Table 2:
  selected wavelength 1537.78363794 nm, certificate 0.99133054946, and
  yield bound 6.0773739905e-5.
- The 930-point map has 442 nonempty fidelity-certified domains and 226
  points certifying both targets. Physical results are unchanged within the
  comparison tolerance. Corrected interval counts differ in 219 W_q rows and
  233 near-optimal-set rows.
- All 64 scenario classifications, 132 robustness flags, and 16 FWM-profile
  candidate counts match the supplied outputs. The 25 GHz peak summed
  per-traversal FWM occupancy decreases by approximately 8.86% at each anchor.
- The boundary solve has 31 rows, no unresolved cases, and a maximum
  certificate residual of 9.40e-7 against the 1e-6 target.
- All 18 numerical manifests match the delivered `src/` hashes, and all
  declared output paths exist. Main Table 2 includes all five main points;
  Supplementary Table S2 retains the four original convergence anchors.

The `logs/` directory contains the final successful stage logs.
`workflow_status.json` records the stage commands and reruns.
`source_changes.patch` compares the release with public commit
`e271dafa2a92dc38d7ce3d567a9cbcc41155800b`.

`integration_validation.csv` and `full_chain_convergence.csv` come from the
preceding FWM-specific audit; the two FWM integration source files are unchanged
from that audit. It compared 18 line/filter integrals with adaptive quadrature
and eight full-chain cases at quadrature orders 48 and 96. The maximum relative
error in the 18 corrected order-48 integrals was 1.27e-7. The current test suite
also includes independent adaptive-quadrature regression cases.

These checks assess implementation and reproducibility under the declared
model. They do not establish measured receiver performance, thermal noise
statistics, independence in a deployed fibre, or journal acceptance.
