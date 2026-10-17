"""Spontaneous Raman background: span-output spectra with explicit direction,
and calibration against the reported measurement point (Methods: receiver filter and Raman normalization)."""
import numpy as np
from units import (C_LIGHT, nm_to_hz, rho_km_nm_to_per_m_per_hz, bose_occupancy)

# ---- Reported calibration point (Methods: receiver filter and Raman normalization, source [R1]) -----------
# POLARIZATION CONVENTION.  Qi et al. define the Raman power in their Eq. (5),
# convert it to photons per spatial-temporal mode in Eq. (6), and apply a factor
# 1/2 explicitly in Eq. (15) to select ONE polarization.  Within their
# convention the quantity carried by the coefficient below is therefore
# POLARIZATION-SUMMED.  The consistent implementation here is: build the
# polarization-summed spectrum with this coefficient, then apply the factor 1/2
# exactly once in `modes.occupancy_from_spectrum` to obtain occupancy per
# polarization.  Do NOT halve a quantity already expressed per polarization.
# Whether a real link has equal H and V occupancy is a separate physical
# assumption, stated in the configuration.
QI_RHO_KM_NM = 2.85e-9        # km^-1 nm^-1, as reused in the literature
QI_PUMP_NM = 1559.79
QI_OBS_NM = 1554.94
# delta_f = f_classical - f_observed  (negative => anti-Stokes)
QI_DELTA_F_HZ = nm_to_hz(QI_PUMP_NM) - nm_to_hz(QI_OBS_NM)


# ---- Hollenbeck & Cantrell (2002) intermediate-broadening model ------------
# Thirteen vibrational modes fitted to the MEASURED Raman gain spectrum of fused
# silica: D. Hollenbeck and C. D. Cantrell, J. Opt. Soc. Am. B 19, 2886 (2002),
# Table 1.  Columns: component position (cm^-1), peak intensity, Gaussian FWHM
# (cm^-1), Lorentzian FWHM (cm^-1).  The response function is
#
#     h(t) = sum_i A_i exp(-gamma_i t) exp(-Gamma_i^2 t^2 / 4) sin(omega_i t),
#
# with omega_i = 2 pi c nu_i, Gamma_i = pi c G_i and gamma_i = pi c L_i, and the
# gain spectrum is proportional to Im H(Omega), H the Fourier transform of h.
# The model describes STIMULATED gain; the conversion to a spontaneous
# background (Bose factor on each side) is applied in `rho_shape`.
_HC_TABLE = np.array([
    # nu (cm^-1)  A_i     G (cm^-1)  L (cm^-1)
    [56.25,       1.00,   52.10,     17.37],
    [100.00,     11.40,  110.42,     38.81],
    [231.25,     36.67,  175.00,     58.33],
    [362.50,     67.67,  162.50,     54.17],
    [463.00,     74.00,  135.33,     45.11],
    [497.00,      4.50,   24.50,      8.17],
    [611.50,      6.80,   41.50,     13.83],
    [691.67,      4.60,  155.00,     51.67],
    [793.67,      4.20,   59.50,     19.83],
    [835.50,      4.50,   64.30,     21.43],
    [930.00,      2.70,  150.00,     50.00],
    [1080.00,     3.10,   91.00,     30.33],
    [1215.00,     3.00,  160.00,     53.33],
])
_HC_CACHE = {}


def _hc_gain_table(df_max_thz=45.0, step_thz=0.002, t_max=20e-12, dt=0.5e-15):
    """Tabulate Im H(Omega) of the Hollenbeck-Cantrell response on a frequency
    grid by direct sine-transform quadrature of h(t).  The table is built once
    and interpolated afterwards; the time grid resolves the fastest mode
    (period ~27 fs) and extends well beyond the slowest decay (~1.3 ps)."""
    key = (df_max_thz, step_thz, t_max, dt)
    if key in _HC_CACHE:
        return _HC_CACHE[key]
    cm = 100.0 * C_LIGHT                       # (cm^-1) -> s^-1 per unit 2 pi
    t = np.arange(0.0, t_max, dt)
    h = np.zeros_like(t)
    for nu, A, G, Lw in _HC_TABLE:
        omega = 2.0 * np.pi * cm * nu
        Gam = np.pi * cm * G
        gam = np.pi * cm * Lw
        h += A * np.exp(-gam * t) * np.exp(-0.25 * (Gam * t) ** 2) * np.sin(omega * t)
    df = np.arange(0.0, df_max_thz + step_thz / 2, step_thz) * 1e12
    im = np.empty_like(df)
    for k0 in range(0, len(df), 512):          # chunked to bound memory
        Om = 2.0 * np.pi * df[k0:k0 + 512]
        im[k0:k0 + 512] = np.trapezoid(h[None, :] * np.sin(Om[:, None] * t[None, :]),
                                       t, axis=1)
    im = np.clip(im, 0.0, None)                # gain is nonnegative for df >= 0
    _HC_CACHE[key] = (df, im / im.max())
    return _HC_CACHE[key]


def _lowfreq_cut(x_thz, x_cut=0.8):
    """Enforce g(0)=0 without distorting the few-THz region excessively."""
    return -np.expm1(-(np.asarray(x_thz, dtype=float) / x_cut) ** 2)


def gain_shape(delta_f_hz, model="hollenbeck_cantrell_2002"):
    """Dimensionless silica Raman gain shape g(|delta_f|).

    `hollenbeck_cantrell_2002` is the published intermediate-broadening fit to
    the measured silica gain spectrum (literature model, used in the main
    results).  `multi_gaussian_v1` and `damped_oscillator_v1` are the earlier
    stand-in shapes; they are not fits to any measured spectrum and are kept
    only as model-sensitivity scenarios.
    """
    x = np.abs(np.asarray(delta_f_hz, dtype=float)) / 1e12      # THz
    if model == "hollenbeck_cantrell_2002":
        df_tab, g_tab = _hc_gain_table()
        return np.interp(x * 1e12, df_tab, g_tab, right=0.0)
    if model == "multi_gaussian_v1":
        peaks = [(0.30, 4.4, 2.0), (0.55, 9.5, 2.5), (1.00, 13.2, 1.6),
                 (0.55, 15.5, 2.2), (0.30, 18.5, 2.0), (0.12, 23.5, 2.5),
                 (0.10, 26.5, 2.0)]
        g = sum(a * np.exp(-0.5 * ((x - c) / w) ** 2) for a, c, w in peaks)
        return g * _lowfreq_cut(x)
    if model == "damped_oscillator_v1":
        x0, gam = 13.2, 10.0
        return (2.0 * x0 * gam * x) / ((x0 ** 2 - x ** 2) ** 2 + (gam * x) ** 2)
    raise ValueError(f"unknown Raman shape model {model!r}")


def thermal_factor(delta_f_hz, temperature_k=300.0):
    """Stokes (delta_f > 0) carries n_B+1; anti-Stokes (delta_f < 0) carries n_B."""
    d = np.asarray(delta_f_hz, dtype=float)
    nb = bose_occupancy(d, temperature_k)
    return np.where(d >= 0.0, nb + 1.0, nb)


def rho_shape(delta_f_hz, model="hollenbeck_cantrell_2002", temperature_k=300.0):
    return gain_shape(delta_f_hz, model) * thermal_factor(delta_f_hz, temperature_k)


def calibrated_rho(delta_f_hz, model="hollenbeck_cantrell_2002", temperature_k=300.0,
                   scale=1.0, rho_km_nm=QI_RHO_KM_NM, lam_obs_nm=QI_OBS_NM,
                   delta_f_meas_hz=QI_DELTA_F_HZ):
    """rho_R(delta_f) in m^-1 Hz^-1, normalized at the measurement's ACTUAL
    frequency separation:

        rho_R(df) = rho_meas * g(df) / g(df_meas)

    `rho_meas` is converted from km^-1 nm^-1 with the spectral Jacobian taken at
    the measurement's observation wavelength.
    """
    rho_meas_f = rho_km_nm_to_per_m_per_hz(rho_km_nm, lam_obs_nm)
    g_meas = rho_shape(delta_f_meas_hz, model, temperature_k)
    return scale * rho_meas_f * rho_shape(delta_f_hz, model, temperature_k) / g_meas


def _long_factor(x, L):
    """(1 - exp(-x L)) / x, stable as x -> 0 (limit L)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x * L) < 1e-8
    safe = np.where(small, 1.0, x)
    out = np.where(small, L, -np.expm1(-safe * L) / safe)
    return out


def span_spectrum(f_obs, dep, a_obs, a_ci, L_m, copropagating,
                  model="hollenbeck_cantrell_2002", temperature_k=300.0, scale=1.0):
    """Span-output Raman PSD (W/Hz, summed over H and V) at observation
    frequencies `f_obs`, BEFORE any downstream component loss.

    co       : S = P0 rho e^{-a_obs L} (1-e^{-[a_ci-a_obs]L}) / (a_ci - a_obs)
    counter  : S = P0 rho             (1-e^{-[a_ci+a_obs]L}) / (a_ci + a_obs)
    """
    f_obs = np.atleast_1d(np.asarray(f_obs, dtype=float))
    a_obs = np.atleast_1d(np.asarray(a_obs, dtype=float))
    fc = dep.freqs_hz[None, :]
    P0 = dep.launch_w[None, :]
    aci = np.atleast_1d(np.asarray(a_ci, dtype=float))[None, :]
    ao = a_obs[:, None]
    delta = fc - f_obs[:, None]
    rho = calibrated_rho(delta, model, temperature_k, scale)
    if copropagating:
        long = _long_factor(aci - ao, L_m) * np.exp(-ao * L_m)
    else:
        long = _long_factor(aci + ao, L_m)
    return np.sum(P0 * rho * long, axis=1)


def longitudinal_check(a_ci, a_obs, L_m, copropagating, n=200001):
    """Direct longitudinal integration of the generation/propagation product,
    used to verify the closed forms."""
    z = np.linspace(0.0, L_m, n)
    if copropagating:
        integ = np.exp(-a_ci * z) * np.exp(-a_obs * (L_m - z))
    else:
        integ = np.exp(-a_ci * z) * np.exp(-a_obs * z)
    return float(np.trapezoid(integ, z))


# ---- Independent near-carrier measurement -----------------------------------
# T. F. da Silva, G. B. Xavier, G. P. Temporao and J. P. von der Weid, J.
# Lightwave Technol. 32, 2332 (2014), Table II: spontaneous Raman coefficient
# slopes measured in SMF-28 across the C band, per 100 GHz of carrier offset,
# for a 10 GHz detection bandwidth, in units of 1e-12 km^-1.  The paper labels
# the two sides "Stokes" and "anti-Stokes"; the larger slope is assigned to the
# anti-Stokes side there, which is the opposite of what detailed balance
# predicts, so the comparison below uses the four-value average and treats the
# side assignment as unresolved.
DASILVA_SLOPES_1E12_PER_KM = {"co_stokes": 6.9, "co_antistokes": 11.5,
                              "counter_stokes": 6.8, "counter_antistokes": 10.8}
DASILVA_DETECTION_BW_HZ = 10e9
DASILVA_MAX_OFFSET_HZ = 1.4e12        # largest carrier offset in their Table I


def dasilva_rho(delta_f_hz, slope_1e12_per_km=None):
    """Near-carrier spontaneous Raman coefficient implied by the measured linear
    slopes, in m^-1 Hz^-1 (both polarizations, as measured).

        rho_f(df) = slope * (|df| / 100 GHz) / B_det,

    valid only over the measured offsets (0.1 - 1.4 THz)."""
    if slope_1e12_per_km is None:
        slope_1e12_per_km = float(np.mean(list(DASILVA_SLOPES_1E12_PER_KM.values())))
    df = np.abs(np.asarray(delta_f_hz, dtype=float))
    beta_per_km = slope_1e12_per_km * 1e-12 * (df / 100e9)
    return beta_per_km / 1e3 / DASILVA_DETECTION_BW_HZ
