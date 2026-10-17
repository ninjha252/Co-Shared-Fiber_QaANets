"""Classical filter leakage with path-specific transfer (Methods: filter leakage).

The remote path (classical transmitter at the far terminal) traverses the span;
the local path (transmitter co-located with the quantum receiver) does not.
Terminal isolation and the wavelength-selective filter are specified separately
so neither is applied twice.
"""
import numpy as np
from units import H_PLANCK


def modulation_lineshape(f, f_c, rms_hz):
    """Normalized Gaussian modulation/linewidth spectrum, integral 1 over f."""
    s = float(rms_hz)
    return np.exp(-0.5 * ((np.asarray(f, float) - f_c) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def incident_spectrum(f, dep, atten, L_m, local, mod_rms_hz,
                      broadband_floor_per_hz, local_tap_transmission):
    """Classical PSD (W/Hz, both polarizations) incident at the receiving
    terminal's filter input, BEFORE the filter and terminal isolation.

    `broadband_floor_per_hz` is the transmitter's out-of-band emission expressed
    as a fraction of carrier power per Hz.
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    fc = dep.freqs_hz[None, :]
    P0 = dep.launch_w[None, :]
    shape = modulation_lineshape(f[:, None], fc, mod_rms_hz) + broadband_floor_per_hz
    if local:
        path = local_tap_transmission
    else:
        lam_nm = 299792458.0 / dep.freqs_hz * 1e9
        path = np.exp(-atten.a_per_m(lam_nm) * L_m)[None, :]
    return np.sum(P0 * shape * path, axis=1)


def system_transfer(f, filt, isolation_extra_db):
    """T_sys(f) = T_q(f) * (extra path isolation).  The filter rejection is
    contained in T_q and is applied exactly once."""
    return np.asarray(filt(f), dtype=float) * 10.0 ** (-abs(isolation_extra_db) / 10.0)
