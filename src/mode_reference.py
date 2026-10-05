"""Independent full-chain spectral reference for the declared receiver
(Methods: receiver model).

The production model reduces every traversal to a scalar pair (eta_m, nu_m) and
composes them with

    T = prod_m eta_m,     nu = eta_3 eta_2 nu_1 + eta_3 nu_2 + nu_3.

This module recomputes the same two quantities WITHOUT that reduction, by
keeping the full frequency dependence of the field transfer through the whole
chain:

    t_tot(f) = prod_m t_m(f),
    T_ref    = | int psi_out^*(f) t_tot(f) psi_in(f) df |^2,
    nu_ref   = sum_m int |psi(f)|^2 [prod_{l>m} |t_l(f)|^2]
                     S_add,out^(m)(f) / (2 h f) df.

`S_add,out^(m)` is the newly added polarization-summed background at the OUTPUT
plane of traversal m, so it already contains the losses of that traversal and
must be multiplied only by the transfers that follow it.

Assumptions (stated, not derived here): stationary independent additive
Gaussian backgrounds, vacuum in the unused input signal modes, no frequency
conversion in the propagation/filter chain, passive polarization operations,
and identical fixed input/output modes with compensated spectral phase.  This
is a numerical reference for the declared receiver, not a general multimode
model.

Nothing in this module calls `optical_link.evaluate_candidate`; the spectra are
rebuilt here from `raman`, `fwm` and `leakage` so the comparison is independent
of the reduction it checks.
"""
import numpy as np

import raman as rm
import leakage as lk
from modes import GaussianMode, SuperGaussianFilter
from units import C_LIGHT, H_PLANCK


def _build(sc, f_q):
    mode = GaussianMode(f_q, sc.mode_fwhm_hz)
    filt = SuperGaussianFilter(f_q, sc.filter_fwhm_hz, sc.filter_order,
                               sc.filter_isolation_db, sc.modeled_domain_hz)
    return mode, filt


def _amp_traversal(sc, filt, f):
    """Field transfer of one complete traversal, including its own launch and
    receive insertion amplitudes, the span, and the receive filter."""
    lam_nm = C_LIGHT / np.asarray(f, dtype=float) * 1e9
    a = sc.atten.a_per_m(lam_nm, allow_extrapolation=True)
    return (sc.eta_tx * sc.eta_rx * np.exp(-a * sc.length_m)
            * filt(f)) ** 0.5


def full_chain_reference(sc, f_q, e0_per_pass=None, sigma2_phi=None, order=None):
    """Return the full-chain (T_ref, nu_ref) and the scalar-recurrence values."""
    import optical_link as ol            # only for the scalar comparison values
    order = int(order or sc.gh_order)
    mode, filt = _build(sc, f_q)
    nodes, w = mode.gauss_hermite(order)

    t1 = _amp_traversal(sc, filt, nodes)
    t_tot = t1 ** 3                      # the three traversals share the model
    T_ref = float(np.abs(np.sum(w * t_tot)) ** 2)

    a_ci = sc.a_classical()
    nu_ref = 0.0
    nu_ref_parts = np.zeros(3)
    prods = sc.fwm_products() if sc.enable_fwm else None
    for m in range(3):
        co = sc.dep.copropagating(m)
        local = sc.dep.local_transmitter(m)
        # transfers AFTER traversal m, in power
        follow = np.ones_like(nodes)
        for _ in range(2 - m):
            follow = follow * np.abs(t1) ** 2

        a_obs = sc.atten.a_per_m(C_LIGHT / nodes * 1e9, allow_extrapolation=True)
        S = rm.span_spectrum(nodes, sc.dep, a_obs, a_ci, sc.length_m, co,
                             sc.raman_model, sc.temperature_k, sc.raman_scale)
        S_out = S * sc.eta_rx * filt(nodes)
        if sc.enable_leakage:
            iso_db = (sc.leak_isolation_local_db if local
                      else sc.leak_isolation_remote_db)
            S_inc = lk.incident_spectrum(nodes, sc.dep, sc.atten, sc.length_m,
                                         local, sc.mod_rms_hz,
                                         sc.tx_broadband_floor_per_hz,
                                         local_tap_transmission=1.0)
            S_out = S_out + S_inc * lk.system_transfer(nodes, filt, iso_db) * (
                1.0 if local else sc.eta_rx)
        part = 0.5 * float(np.sum(w * follow * S_out / (H_PLANCK * nodes)))

        if prods is not None and sc.dep.quantum_directions[m] == sc.dep.direction:
            win = sc.mode_line_window_sigma * (mode.sigma + sc.fwm_profile_rms_hz)
            lo, hi = np.searchsorted(prods.freq_hz, [f_q - win, f_q + win])
            fl = prods.freq_hz[lo:hi]
            pl = prods.power_incoherent_w[lo:hi]
            if len(fl):
                if sc.fwm_profile_rms_hz > 0:
                    x, wl = np.polynomial.hermite.hermgauss(order)
                    wl = wl / np.sqrt(np.pi)
                    ff = fl[:, None] + np.sqrt(2.0) * sc.fwm_profile_rms_hz * x[None, :]
                    fo = np.ones_like(ff)
                    for _ in range(2 - m):
                        fo = fo * np.abs(_amp_traversal(sc, filt, ff)) ** 2
                    part += 0.5 * float(np.sum(
                        pl[:, None] * wl[None, :] * mode.intensity(ff)
                        * sc.eta_rx * filt(ff) * fo / (H_PLANCK * ff)))
                else:
                    fo = np.ones_like(fl)
                    for _ in range(2 - m):
                        fo = fo * np.abs(_amp_traversal(sc, filt, fl)) ** 2
                    part += 0.5 * float(np.sum(
                        pl * mode.intensity(fl) * sc.eta_rx * filt(fl) * fo
                        / (H_PLANCK * fl)))
        nu_ref_parts[m] = part
        nu_ref += part

    pt = ol.evaluate_candidate(sc, f_q, want_components=False)
    eta = np.asarray(pt["eta"], float)
    nu = np.asarray(pt["nu"], float)
    T_scalar = float(np.prod(eta))
    nu_scalar = float(eta[2] * eta[1] * nu[0] + eta[2] * nu[1] + nu[2])

    out = dict(wavelength_nm=C_LIGHT / f_q * 1e9, frequency_hz=float(f_q),
               receiver_mode_convention=sc.receiver_mode_convention,
               T_ref=T_ref, T_scalar=T_scalar,
               T_relative_difference=(T_ref - T_scalar) / T_scalar,
               nu_ref=float(nu_ref), nu_scalar=nu_scalar,
               nu_relative_difference=((nu_ref - nu_scalar) / nu_scalar
                                       if nu_scalar > 0 else np.nan),
               nu_ref_per_traversal_contribution=nu_ref_parts.tolist())

    if e0_per_pass is not None and sigma2_phi is not None:
        import local_protocol as lp
        import optical_channel as oc
        import reliability as rl
        z = lp.z_from_sigma2(np.asarray(sigma2_phi, float))
        tw = [lp.traversal_weights(float(e0_per_pass), float(z[m])) for m in range(3)]
        bvec = np.array([w[0] for w in tw])
        for tag, Tv, nuv in (("scalar", T_scalar, nu_scalar),
                             ("ref", T_ref, float(nu_ref))):
            acc = oc.accepted_optical(Tv, nuv)
            q = float(lp.affected_probability(float(acc.wI), bvec))
            C = float(rl.correctable_weight(q, 7, 1))
            out[f"s_{tag}"] = float(acc.s)
            out[f"q_{tag}"] = q
            out[f"C_{tag}"] = C
            out[f"Y_lb_{tag}"] = float(acc.s) ** 7 * C
        out["C_absolute_difference"] = out["C_ref"] - out["C_scalar"]
        out["Y_lb_relative_difference"] = (
            (out["Y_lb_ref"] - out["Y_lb_scalar"]) / out["Y_lb_scalar"]
            if out["Y_lb_scalar"] > 0 else np.nan)
    return out
