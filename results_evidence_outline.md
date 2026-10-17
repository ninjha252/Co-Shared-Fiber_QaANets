# Results evidence outline

One short paragraph per main figure, using only quantities from the regenerated
outputs.  Numerical verification and model-conditional prediction are kept
apart.

**Figure 1 (verification).**  An independent thermal-environment dilation
reproduces the derived accepted optical operation over 68 synthetic
(T, nu) points: maximum accepted-Choi trace-norm error 5.7e-16 against a
1e-9 target, maximum normalized-channel error 1.1e-15 against 1e-7, and
acceptance independent of the input state to 0e+00 over six polarization
probe states.  Three explicit Fock-space traversals with final-only acceptance
reproduce the single effective channel to 2e-16.  These are statements
about the mathematics of the reduction, not about a physical fibre.

**Figure 2 (verification).**  With an explicit Steane encoder and fixed CSS
Hamming recovery, the exact conditional logical entanglement fidelity satisfies
`F_e >= C_7,1(q)` at every tested point on both noise paths; the largest gap is
0.438 and the most negative residual -1e-15.  Two independent
exact methods agree to 1e-13.  Key dependence is small on the baseline path
(5e-05 over 16 random commuting key sets) and significant on the
strong-XPM path (0.164; the fixed non-Clifford set differs from identity
keys by 0.108), which is why the general density-matrix calculation is
used.  This demonstrates key dependence; it does not quantify Pauli-twirl error.

**Figure 3 (model-conditional).**  At 5 km per traversal and -10 dBm per
classical channel, with a 50 GHz guard and 1 GHz added margin,
454 of 454 allowed candidates are certified by the
lower bound.  The constrained maximum-`Y_lb` rule selects 1551.3273 nm
with `s = 0.2880`, `q = 0.01968`, `C = 0.992384`,
`Y_lb = 1.6321e-04` and exact identity-key `F_e = 0.994111`
(`Y = 1.6350e-04`).  Maximum transmission selects 1551.3273 nm (C = 0.992384, the same wavelength; certified for F_min and Y_min); Minimum q selects 1537.0110 nm (C = 0.993576, a different wavelength; certified for F_min and Y_min).  The predeclared fixed reference
1550 nm is INVALID: the reference wavelength is excluded by the guard and margin rule at this deployment.  FWM
products lie on the classical grid, with 92% of the modeled product
power on occupied carriers and 6.0% on the 9 vacant in-band
slots; only vacant positions can coincide with an allowed candidate.  All of
this is conditional on the declared Raman shape, receiver model and background
assumptions.

**Figure 4 (model-conditional).**  Over a sampled map of 930 length-power
points, 442 have a nonempty quality-certified domain and 226 of those
also meet `Y_min`; the smallest optimized `Y_lb` among them is 3.16e-14.
At 128 of the 488 uncertified points the upper bound
`max s^7` still exceeds `Y_min`, so the bounds leave those points unresolved
rather than excluding them.  The fidelity-feasibility boundary is resolved to
9.4e-07 in `C`.  Requiring tuning margin moves the selected wavelength
and changes the optimized yield by the amounts in Supplementary Table S2;
2 enumerated scenarios empty the certified domain at 5 km /
-10 dBm/ch, and the intersection of the certified domains over all scenarios is
empty, so no single wavelength is certified
across the whole enumerated set.

**What these results do not establish.**  They do not show that the physical
assumptions hold in deployed fibre; they do not make any accepted logical state
known to be correct; `T^7` is not an upper bound once background replacement is
included; an empty certified set does not prove physical impossibility; the
lower-bound optimum need not be the optimum of exact yield; and no quantity
reported here is a secret-key rate.
