# Corrected classical fiber-input budget

Review package, 19 October 2026 UTC. GitHub has not been updated.
Base: commit `0553fd0f24a22fea7a17702aaca44aceb16fb7bf`.

## What was corrected

The optical-noise calculation takes classical powers at the fiber input,
after the input multiplexer. The earlier budget also added that upstream
multiplexer's loss. The corrected minimum is

`P_c,min(L) = S_rx + alpha_c L + IL_demux + M_sys`,

with output demultiplexer loss 3 dB and system margin 3 dB. The upstream
transmitter requirement is `P_tx,min = P_c,min + IL_mux`, with input mux loss
3 dB. No extra allowance or 6 dB design margin is used.

The existing numerical model, carrier allocation, grid, receiver model,
Steane recovery, noise formulas, and quadrature settings were retained.
Only the power budget and its dependent calculations were corrected.

## Rerun and results

The existing `scripts/run_link_budget.py` was rerun: 12 operating points,
60 receiver/distance curve rows, and 60 base-error anchor rows. The existing
figure, table and report scripts were then executed. All 36 regression tests
passed, including the new receiver-power balance test.

| Quantity on the existing 1 km grid | Earlier result | Corrected result |
|---|---:|---:|
| 1.25 Gb/s: last sampled fidelity-certified distance | 26 km | 30 km, end of scan |
| 1.25 Gb/s: last sampled distance meeting both targets | 10 km | 10 km |
| 10.7 Gb/s: last sampled fidelity-certified distance | 9 km | 13 km |
| 10.7 Gb/s: last sampled distance meeting both targets | 9 km | 10 km |

These are sampled distances, not continuously refined boundaries. Fidelity
certification at 30 km for the 1.25 Gb/s receiver does not establish where it
eventually fails beyond the scan. The 10.7 Gb/s receiver sensitivity remains
the explicitly assumed -24 dBm benchmark.

At 10 km the corrected powers are -28.9213 dBm per channel for 1.25 Gb/s and
-16.1213 dBm per channel for 10.7 Gb/s. The corrected optimized yield bounds
are 2.066054e-6 and 2.156186e-6, respectively. At 11 km they are 8.648754e-7
and 9.158222e-7. Both curves therefore cross the 1e-6 yield threshold between
the sampled 10 and 11 km points. For 10.7 Gb/s, no wavelength was fidelity
certified at 10 km in the earlier run; the corrected run does certify one.

Lower classical power need not increase the numerical yield bound at every
point: background photons can contribute to accepted one-photon events.
Fidelity and acceptance must be assessed separately under this receiver model.

## Verification and provenance

- All 60 operating-curve powers decrease by 3 dB, within floating-point error.
- Evaluating the unrounded budget and subtracting span and demultiplexer
  losses gives receiver sensitivity plus 3 dB, with zero floating-point
  residual in the explicit check at all 60 points.
- The four predeclared fixed-power cases reproduce their earlier selection,
  certificate, yield, and base-error results within relative tolerance 1e-11.
- Main Table 2 is numerically unchanged.
- 122 numerical CSV files outside the dependent run/table directories remain
  byte-for-byte unchanged, including the fixed-power maps and wavelength scans.
- `link_budget_before_after.csv` contains all 60 old/new curve rows.
- `link_budget_correction.json` records checks, run scope, and software versions.
- `link_budget_source_changes.patch` records the source changes.
- `link_budget_source_and_config_sha256.json` records the current source/config hashes.

The numerical manifest records source hashes at execution start. Subsequently,
only the provenance-description text in `src/runio.py` was corrected; its
executed snapshot is retained in `link_budget_execution_source/runio.py` and
matches the manifest hash. All other numerical source hashes match. Earlier
stage manifests and verification files are historical records of the previous
release, not evidence that those stages were rerun during this correction.
The requirements file records the environment used for this targeted rerun
and current exports; earlier manifests retain their own software versions.

## Figures to use

Use the manuscript PDFs in `results/figures/final/`.

- Figure 4: corrected classical-budget overlays; the underlying fixed-power
  maps, tuning-clearance comparison and sensitivity data are unchanged.
- Figure S2: corrected overlays and recalculated base-error/yield curves.
- Figure S1: clarified FWM occupancy axis; no new S1 simulation was required.
- Figures 1-3: supplied for a complete figure set, rendered from their existing
  data. Their simulation results are unchanged.

All three existing rendering variants were regenerated. Font rendering and
PDF internals may differ with the recorded plotting environment. The files
do not include manuscript captions: those remain in the LaTeX source.

## Exact manuscript updates

1. Replace Equation (1)'s body with the following, preserving its label:

```latex
P_{c,\min}(L)=S_{\mathrm{rx}}+\alpha_cL
+IL_{\mathrm{demux}}+M_{\mathrm{sys}}.
```

2. In Table 1 use:

```latex
$IL_{\mathrm{demux}}=3$~dB,
$M_{\mathrm{sys}}=3$~dB,
$\alpha_c=0.188$~dB km$^{-1}$.
```

3. Remove the additional allowance and 6 dB design-margin explanation. Use:

```latex
Classical powers are specified at the fiber input.
The classical budget includes a $3$~dB demultiplexer loss
and a $3$~dB system margin. Input multiplexer loss determines
the required upstream transmitter power and is excluded
from the fiber-input budget.
```

4. Replace `P_{c,\mathrm{design}}` and the misspelling `desgin` by
`P_{c,\min}`, including Equation (59). Replace references to an adopted
6 dB design curve with the classical link-budget curve. This minimum is
conditional on the stated receiver sensitivity, losses and prescribed margin.

5. For the distance claim in Abstract, Results, Discussion and Appendix C use:

```latex
Along the $1.25$~Gb/s classical link-budget curve at
$e_0=0.2\%$, the optimized yield bound falls below
$10^{-6}$ between $10$ and $11$~km per traversal,
while fidelity remains certified through the largest
evaluated distance of $30$~km.
```

For the high-rate comparison, if reporting numerical reach, use:

```latex
For the assumed $10.7$~Gb/s receiver benchmark,
fidelity remains certified through $13$~km on the
evaluated distance grid, whereas both thresholds
are met only through $10$~km.
```

6. Figure 4 caption can retain "with classical link-budget minima" after
replacing the figure with this corrected export. Figure S2 caption can retain
"along the 1.25 and 10.7 Gb/s link-budget minima". These now refer to the
corrected fiber-input budget. Retain the short corrected Figure S1(c) caption.

Keep the Table 2 certification explanation, Raman-unit correction, XPM
clarification, EXFO uncertainty wording/citation, and convergence statement.
Those corrections are independent of this power-budget issue.

The manuscript PDF itself was not edited. This package addresses the
identified power-reference inconsistency; it does not constitute experimental
validation or a guarantee that no other scientific issue exists.
