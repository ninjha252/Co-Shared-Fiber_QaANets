# Figure captions

Standalone captions.  Every setting named here is the one the run actually used;
every number is read from the stored outputs.

**Figure 1 | Optical-channel verification.**  Ideal final one-photon acceptance
on two polarization modes, each carrying an independent isotropic thermal
background, on synthetic (T, nu) pairs.  (a) Acceptance `s` and (b) conditional
mixing `r` against the per-polarization final occupancy nu at T = 0.05, 0.2 and
0.9.  Curves are the closed-form weights `c0 = T/(1+nu)^4` and
`c1 = 2 nu (1+nu-T)/(1+nu)^4`; markers are an independent thermal-environment
beam-splitter dilation, built exactly inside each total-photon-number sector with
both truncated environments traced out before projection.  The pure-loss limit
nu -> 0 gives `s = T` and `r = 0` exactly and cannot appear on a logarithmic
horizontal axis.  (c) Accepted-Choi trace-norm discrepancy
`||J_num - J_formula||_1` against the discarded thermal probability for four
(T, nu) pairs, each with its own colour and marker shape.  The **red star on
each series marks the production setting** (environment tail tolerance 1e-16),
which every reported run uses; the remaining points are deliberately loosened
cutoffs, shown only to exhibit the error tracking the discarded tail.  The
**black dashed line is the 1e-9 target tolerance**.  Over all
68 sweep points the achieved discrepancy is 5.7e-16 and the
normalized-channel error is 1.1e-15.  The three-traversal composition
check and the receiver-model comparison are in Supplementary Figure S1.  An
independent numerical construction reproduces the adopted accepted optical
channel over the tested domain; no claim about measured fibre noise follows.

**Figure 2 | Logical fidelity and the correction bound.**  [[7,1,3]] Steane code
with the fixed CSS Hamming recovery.  Both noise paths are one-parameter
families indexed by u (0 plus 33 logarithmic values from 1e-3 to 10) at fixed
transmission eta = 0.55 per pass; variances are scaled first and converted to z
afterwards.  *Baseline* at u = 1: nu = 3e-4 per polarization, e_0 = 0.002,
sigma_phi^2 = 4e-5 on the two co-propagating passes and 0 on the
counter-propagating pass.  *Strong XPM* at u = 1: nu = 3e-5, e_0 = 2e-4,
sigma_phi^2 = 0.10 on all three passes.  (a) Exact logical infidelity `1 - F_e`
against the affected-location probability q on the baseline path, with
`1 - C_7,1(q)` as the upper bound on the infidelity; identity and fixed
non-Clifford keys `A_j = R_y(0.628)`, `B_j = R_y(1.047)` are density-matrix
results and the circles are an independent enumeration of all 4^7 net Pauli
strings for identity keys.  (b) Bound gap `F_e - C_7,1(q)` on both paths (inset:
baseline only, logarithmic).  The gap is nonnegative everywhere, maximum
0.438, most negative value -1e-15; it reflects support counting,
cancellation inside affected histories, and errors outside the guaranteed set
that the recovery also corrects.  (c) Sampled key dependence: the range
max-min over 16 reproducible commuting random key sets (markers) and the
absolute difference between the fixed non-Clifford and identity cases (dashed);
the q = 0 point carries no key dependence and is omitted.  The shaded band is the
measured enumeration versus density-matrix residual (1e-13), a numerical
reference, not an error bar for every non-Clifford computation.  The baseline
range reaches 5e-05; the strong-XPM range reaches 0.164 and the
fixed non-Clifford case differs from identity keys by up to 0.108.  The
random key rule draws a common axis per location from x, y, z and both angles
from [0, 2 pi): not Haar sampling, so the sampled range is not a worst-case
guarantee over private keys, and key dependence alone does not measure
Pauli-twirl error.

**Figure 3 | Wavelength selection.**  Predeclared operating point: 5 km per
traversal, -10 dBm per classical channel; 35 fixed carriers on a 100 GHz grid
from 191 600 to 195 900 GHz launched A->B, so the three quantum traversals are
co-, counter- and co-propagating.  Guard 50 GHz with an added tuning
margin of 1 GHz (a declared scenario choice, not a hardware
measurement); candidates on a 2 GHz grid with
the exact hardware, guard and search boundary points inserted; accepted mode
Gaussian of 10 GHz FWHM; receiver filter
super-Gaussian of order 4,
25 GHz FWHM and
110 dB floor; insertion transmission
0.82 per pass split symmetrically; base
error 0.002 per pass; fixed-mode receiver convention.
(a) Final propagated Raman, FWM and leakage occupancies PER POLARIZATION, each
attenuated by the later traversals; FWM components are delta lines (narrow-line
limit).  (b) Single-mode
acceptance `s`: its peaks are background replacement photons, not improved
quantum transmission.  (c) The lower-bound certificate `C_7,1(q)` and the exact
logical fidelity `F_e` for identity keys, computed by Pauli enumeration at every
allowed candidate, against `F_min = 0.99`.  Certification under the lower
bound is a different statement from the exact fidelity of one key set.
(d) `Y_lb`, `Y_ub = s^7` and the exact `Y = s^7 F_e` for identity keys, with
`Y_min = 1e-06`; the colours pair `Y_lb` with `C` and `Y` with `F_e` in (c).
The shaded bands are the certified region `W_q` (3 finite-width
components, 0 zero-width isolated
points; isolated points would be drawn as thin
stems).  Vertical guide lines through all panels mark the comparator
selections, and coincident selections share one line.  Here the
constrained maximum-`Y_lb` rule selects 1551.3273 nm with
`s = 0.2880`, `C = 0.992384`, `Y_lb = 1.6321e-04` and exact
`F_e = 0.994111`.  The selected wavelength maximizes the LOWER BOUND over
the certified set; it need not maximize the exact yield.  Selector agreement is
reported as found: Maximum transmission selects 1551.3273 nm (C = 0.992384, the same wavelength; certified for F_min and Y_min); Minimum q selects 1537.0110 nm (C = 0.993576, a different wavelength; certified for F_min and Y_min).  The predeclared fixed reference
1550 nm is INVALID: the reference wavelength is excluded by the guard and margin rule at this deployment; the
nearest allowed candidate (1551.3273 nm) is a distinct rule.
The Raman spectrum is the published Hollenbeck-Cantrell fit to measured silica
gain, normalized at the measured coefficient of Qi et al.; the aggregate
background is assumed isotropic thermal.

**Figure 4 | Operating regions and sensitivity.**  Same fixed classical
allocation, receiver assumptions and targets; the quantum wavelength is
re-optimized at every point, with guard 50 GHz and 1 GHz added
margin in panels (a), (b) and (d).  (a) `log10 max_Omega_F Y_lb` over 30 lengths
by 31 launch powers on a 1 km by 1 dB sampled grid; grey marks the
488 of 930 sampled points where the quality-certified domain is
empty.  Undefined optima are masked, never zero-filled.
A Y_min contour is drawn where optimized yields cross it.
The black curve is the fidelity-feasibility boundary `max_lambda C = F_min`,
located by bracketed bisection on directly re-evaluated maxima (worst residual
9.4e-07 in `C`; monotonicity in length is not assumed).  The dashed
and dotted black lines are the minimum launch power that closes the classical
link for the 1.25 Gb/s and 10.7 Gb/s
receivers (sensitivity, worst classical attenuation, multiplexer, demultiplexer
and margin); points above a line close that link.  (b) Selected wavelength on
the same plane, masked where the optimum is undefined, with the same link-budget
lines.
(c) Tuning-margin sensitivity at the predeclared interior point (3 km,
-13 dBm/ch) and the original comparison point (5 km, -10 dBm/ch): wavelength
displacement (upper) and relative change in the optimized `Y_lb` (lower) for
margins 0, 1, 2, 2.5 and 5 GHz at the 50 GHz guard.  Ratios
are computed only where both compared optima exist; a red cross marks a margin
at which the certified set is empty.  (d) Scenario sensitivity at 5 km,
-10 dBm/ch: bars are `M_theta = max_(s>0) C_7,1(q)` with the wavelength free,
red exactly when `M_theta < F_min`, which means no candidate is certified by this
bound over the evaluated domain, NOT that exact fidelity fails; open diamonds are
the certificate at the nominal wavelength held fixed.  Scenario ranges are
declared sensitivity choices, not confidence intervals.

**Supplementary Figure S1 | Composition check, receiver-model comparison and FWM
profile widths.**  (a) Trace-norm discrepancy between three explicit Fock-space
traversals with final-only acceptance and the single effective (T, nu) channel.
(b) Relative difference of the full-chain spectral reference from the scalar
recurrence, (full chain / scalar) - 1 in percent, for `T`, `nu` and `Y_lb` at
the evaluated candidates of the four predeclared points;
the main results use the scalar recurrence with the declared mode restoration at
each traversal output, and this panel is the size of that modelling choice.
(c) Peak propagated FWM occupancy over the allowed domain against the prescribed
profile rms width; the widths are prescribed, not derived from a modulation
model.

**Supplementary Figure S2 | Base-error requirement and the classical link
budget.**  (a) Largest per-traversal base error `e_0` for which some allowed
candidate still meets `C_7,1(q) >= F_min`, over the operating map of Fig. 4;
grey marks points where no candidate is certified even at `e_0 = 0`.  White
contours are the published `e_0` anchors that fall inside the plotted range
(marked on the colour bar), with the same line styles as in (b).  Black lines
are the link-budget minimum launch power of each receiver, as in Fig. 4.
(b) The same quantity along each link-budget line (orange circles, solid line:
1.25 Gb/s; green squares, dashed line: 10.7 Gb/s; markers are staggered so that
coincident curves stay visible).  Grey horizontal lines are the anchors, named on the
right axis: 0.07%, Agnesi et al., Optica 7, 284 (2020): self-compensating encoder, key basis; 0.2%, nominal value used in the main results; 0.4%, Bolanos et al., Optica Quantum 4, 389 (2026): complete GHz-rate system; 0.7%, Xavier et al., New J. Phys. 11, 045015 (2009): 16 km fibre, static; 1.8%, Xavier et al. (2009): 16 km fibre with active compensation running.
A receiver is certified at the nominal `e_0` wherever its curve lies above the
`nominal` line.  (c) The optimized lower yield bound `Y_lb` along each
link-budget line, with the same markers at the nominal `e_0 = 0.2%`,
with `Y_min` dashed.  1.25 Gb/s receiver (-36.8 dBm): fidelity certified up to 26 km and Y_min met up to 10 km; 10.7 Gb/s receiver (-24 dBm): fidelity certified up to 9 km and Y_min met up to 9 km.  At the 5 km,
-10 dBm/ch point the largest admissible `e_0` is 0.305%.
