"""Unit conversions and physical constants. All internal quantities are SI:
metres, seconds, hertz, watts, radians, dimensionless probabilities."""
import numpy as np

C_LIGHT = 299792458.0          # m/s (SI definition)
H_PLANCK = 6.62607015e-34      # J s (SI definition)
K_BOLTZ = 1.380649e-23         # J/K (SI definition)


def dbm_to_w(p_dbm):
    """dBm -> W."""
    return 1e-3 * 10.0 ** (np.asarray(p_dbm, dtype=float) / 10.0)


def w_to_dbm(p_w):
    """W -> dBm."""
    return 10.0 * np.log10(np.asarray(p_w, dtype=float) / 1e-3)


def db_per_km_to_per_m(alpha_db_km):
    """Power attenuation in dB/km -> a in m^-1 such that P(z)=P0 exp(-a z).

    A field AMPLITUDE decays as exp(-a z / 2).
    """
    return np.log(10.0) / (10.0 * 1e3) * np.asarray(alpha_db_km, dtype=float)


def per_m_to_db_per_km(a_per_m):
    return np.asarray(a_per_m, dtype=float) * 10.0 * 1e3 / np.log(10.0)


def nm_to_hz(lam_nm):
    """Vacuum wavelength [nm] -> frequency [Hz]."""
    return C_LIGHT / (np.asarray(lam_nm, dtype=float) * 1e-9)


def hz_to_nm(f_hz):
    """Frequency [Hz] -> vacuum wavelength [nm]."""
    return C_LIGHT / np.asarray(f_hz, dtype=float) * 1e9


def dlambda_df_magnitude(lam_m):
    """|d(lambda)/d(f)| in m/Hz at vacuum wavelength lam_m [m]."""
    lam_m = np.asarray(lam_m, dtype=float)
    return lam_m ** 2 / C_LIGHT


def rho_km_nm_to_per_m_per_hz(rho_km_nm, lam_obs_nm):
    """Raman coefficient reported per km per nm -> per m per Hz.

    rho_f = rho_km_nm / 1e3 * |d lambda_nm / d f|
          = rho_km_nm / 1e3 * 1e9 * lam_m^2 / c
    """
    lam_m = np.asarray(lam_obs_nm, dtype=float) * 1e-9
    return np.asarray(rho_km_nm, dtype=float) / 1e3 * 1e9 * lam_m ** 2 / C_LIGHT


def bose_occupancy(delta_f_hz, temperature_k):
    """Bose-Einstein occupancy n_B(|delta_f|) at temperature T."""
    x = H_PLANCK * np.abs(np.asarray(delta_f_hz, dtype=float)) / (K_BOLTZ * temperature_k)
    # stable for small x: 1/expm1(x)
    with np.errstate(divide="ignore", over="ignore"):
        out = 1.0 / np.expm1(np.maximum(x, 1e-300))
    return out


def photon_energy(f_hz):
    return H_PLANCK * np.asarray(f_hz, dtype=float)
