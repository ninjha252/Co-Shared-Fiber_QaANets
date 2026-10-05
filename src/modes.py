"""Accepted spatial-temporal mode, receiver filter, and mode occupancy
(Methods: accepted mode)."""
import numpy as np
from units import H_PLANCK

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


class GaussianMode:
    """Gaussian spectral INTENSITY |psi(f)|^2 with unit integral.

    |psi(f)|^2 = exp(-(f-f0)^2/(2 sigma^2)) / (sigma sqrt(2 pi)) ,  units 1/Hz.

    For a transform-limited Gaussian the intensity time-bandwidth FWHM product is
    0.4413; `time_fwhm_s` reports the implied pulse duration so that an
    incompatible independent pulse width is never assigned.
    """
    TBP = 0.4413

    def __init__(self, f0_hz, fwhm_hz):
        self.f0 = float(f0_hz)
        self.fwhm = float(fwhm_hz)
        self.sigma = self.fwhm * FWHM_TO_SIGMA

    @property
    def time_fwhm_s(self):
        return self.TBP / self.fwhm

    def intensity(self, f):
        f = np.asarray(f, dtype=float)
        return np.exp(-0.5 * ((f - self.f0) / self.sigma) ** 2) / (
            self.sigma * np.sqrt(2 * np.pi))

    def gauss_hermite(self, order=48):
        """Nodes and weights for  int |psi(f)|^2 G(f) df  ~  sum w_k G(f_k)."""
        x, w = np.polynomial.hermite.hermgauss(order)
        f = self.f0 + np.sqrt(2.0) * self.sigma * x
        return f, w / np.sqrt(np.pi)

    def norm_check(self, order=48):
        f, w = self.gauss_hermite(order)
        return float(np.sum(w))


class SuperGaussianFilter:
    """Receiver power response

        T_q(f) = (1 - floor) exp(-ln2 * (2 (f-f0)/FWHM)^(2 m)) + floor

    over a FINITE modeled optical domain; outside that domain T_q = 0, so the
    equivalent noise bandwidth stays finite (Methods: receiver filter and Raman normalization).
    """

    def __init__(self, f0_hz, fwhm_hz, order=3, isolation_db=110.0,
                 domain_hz=None):
        self.f0 = float(f0_hz)
        self.fwhm = float(fwhm_hz)
        self.order = int(order)
        self.floor = 10.0 ** (-abs(isolation_db) / 10.0)
        self.domain = domain_hz

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        x = 2.0 * (f - self.f0) / self.fwhm
        core = np.exp(-np.log(2.0) * np.clip(x ** (2 * self.order), 0, 700))
        out = (1.0 - self.floor) * core + self.floor
        if self.domain is not None:
            lo, hi = self.domain
            out = np.where((f >= lo) & (f <= hi), out, 0.0)
        return out

    def equivalent_noise_bandwidth(self, n=200001):
        """B_eq = int T_q(f) df over the modeled optical domain.  This is NOT the
        FWHM."""
        if self.domain is None:
            raise ValueError("B_eq requires a finite modeled domain")
        lo, hi = self.domain
        f = np.linspace(lo, hi, n)
        return float(np.trapezoid(self(f), f))


def occupancy_from_spectrum(mode, spectrum_fn, order=48, per_polarization=False):
    """nu = (1/2) int |psi(f)|^2 S(f)/(h f) df  for S summed over H and V.

    Set `per_polarization=True` if the supplied S is already per polarization,
    in which case the factor 1/2 is omitted (Methods: mode occupancy).
    """
    f, w = mode.gauss_hermite(order)
    S = np.asarray(spectrum_fn(f), dtype=float)
    integrand = S / (H_PLANCK * f)
    val = float(np.sum(w * integrand))
    return val if per_polarization else 0.5 * val


def occupancy_from_line(power_w, f_line_hz, mode, transfer=1.0,
                        per_polarization=False):
    """Delta-line contribution:  nu_l = P_l |psi(f_l)|^2 T(f_l) / (2 h f_l).

    Determines the MEAN mode occupancy under the stated polarization
    convention; it does not establish thermal statistics.
    """
    f_line_hz = np.asarray(f_line_hz, dtype=float)
    p = np.asarray(power_w, dtype=float) * np.asarray(transfer, dtype=float)
    val = p * mode.intensity(f_line_hz) / (H_PLANCK * f_line_hz)
    val = float(np.sum(val))
    return val if per_polarization else 0.5 * val


def gate_photon_mean(spectrum_fn, tau_gate_s, f_lo, f_hi, n=20001):
    """mu_gate = tau_g int S(f)/(h f) df  -- a MULTIMODE count, retained only as
    a diagnostic.  It must not be substituted for nu (Methods: accepted mode)."""
    f = np.linspace(f_lo, f_hi, n)
    S = np.asarray(spectrum_fn(f), dtype=float)
    return float(tau_gate_s * np.trapezoid(S / (H_PLANCK * f), f))


def occupancy_from_profiled_lines(mode, power_w, f_line_hz, rms_hz,
                                  transfer_fn=None, order=48,
                                  per_polarization=False):
    """Mode occupancy of generated components carrying a finite spectral profile.

        nu_l = (1/2) P_l int g(f - f_l; sigma_p) |psi(f)|^2 T(f) / (h f) df,

    with g a normalized Gaussian of rms width `rms_hz` and T the transfer from
    the component's declared generation plane to the receiver (receive-side
    transmission times the filter power response).  `rms_hz = 0` reproduces the
    delta-line limit exactly, so the narrow-line result is the analytic limiting
    case of the same expression.

    The profile is PRESCRIBED: it is a phenomenological broadening of each
    generated component at fixed total product power, not a simulation of
    modulated four-wave mixing.  Distinct-channel and degenerate processes carry
    different field statistics and intensity moments; those are not modeled.
    """
    power_w = np.atleast_1d(np.asarray(power_w, dtype=float))
    f_line_hz = np.atleast_1d(np.asarray(f_line_hz, dtype=float))
    if power_w.size == 0:
        return 0.0
    if transfer_fn is None:
        def transfer_fn(f):
            return np.ones_like(np.asarray(f, dtype=float))
    if float(rms_hz) <= 0.0:
        val = float(np.sum(power_w * mode.intensity(f_line_hz)
                           * np.asarray(transfer_fn(f_line_hz), dtype=float)
                           / (H_PLANCK * f_line_hz)))
        return val if per_polarization else 0.5 * val
    x, w = np.polynomial.hermite.hermgauss(int(order))
    w = w / np.sqrt(np.pi)
    # f[l, k] = f_l + sqrt(2) sigma_p x_k
    f = f_line_hz[:, None] + np.sqrt(2.0) * float(rms_hz) * x[None, :]
    integ = (mode.intensity(f) * np.asarray(transfer_fn(f), dtype=float)
             / (H_PLANCK * f))
    val = float(np.sum(power_w[:, None] * w[None, :] * integ))
    return val if per_polarization else 0.5 * val


def mode_overlap_transmission(mode, amplitude_fn, order=48):
    """Fixed-mode modal transmission  eta = |int psi_out^*(f) t(f) psi_in(f) df|^2
    for identical normalized input/output modes, i.e. |int |psi(f)|^2 t(f) df|^2.

    This is NOT the power average int |psi|^2 |t|^2 df: the two agree only for a
    transfer that is flat across the mode.  `amplitude_fn` returns the COMPLEX
    field transfer t(f); a real nonnegative t means the spectral phase is taken
    to be compensated or absorbed into the mode definition.
    """
    f, w = mode.gauss_hermite(order)
    t = np.asarray(amplitude_fn(f), dtype=complex)
    return float(np.abs(np.sum(w * t)) ** 2)


def power_average_transmission(mode, transfer_fn, order=48):
    """Mode-weighted power average  int |psi(f)|^2 T(f) df  (the legacy
    convention, retained for the receiver-model comparison)."""
    f, w = mode.gauss_hermite(order)
    return float(np.sum(w * np.asarray(transfer_fn(f), dtype=float)))
