"""Classical link budget: the smallest launch power that still closes the
classical data link.

The launch power of the classical channels is not a free parameter in a
realistic deployment.  It must be large enough that every data receiver sees at
least its sensitivity after the span and the multiplexing optics, plus an
engineering margin:

    P_launch,min(L) = S_rx + alpha_c L + IL_mux + IL_demux + M,

with S_rx the receiver sensitivity (dBm), alpha_c the worst classical-channel
attenuation (dB/km) and L the span length (km).  Launching more than this only
adds Raman, four-wave-mixing and leakage background to the quantum channel, so
the link-budget minimum is the physically motivated operating power.
"""
import numpy as np


def worst_classical_attenuation_db_km(atten, carrier_wavelengths_nm):
    """Largest attenuation over the classical carriers, in dB/km."""
    from units import per_m_to_db_per_km
    a = atten.a_per_m(np.asarray(carrier_wavelengths_nm, dtype=float))
    return float(np.max(per_m_to_db_per_km(a)))


def minimum_launch_dbm(length_km, rx_sensitivity_dbm, alpha_db_km,
                       mux_db, demux_db, margin_db):
    """Minimum per-channel launch power (dBm) that closes the classical link."""
    L = np.asarray(length_km, dtype=float)
    return (float(rx_sensitivity_dbm) + float(alpha_db_km) * L
            + float(mux_db) + float(demux_db) + float(margin_db))


def receivers(cfg):
    """Receiver definitions from the configuration, each with its source."""
    return list(cfg.get("classical_link", {}).get("receivers", []))


def budget_terms(cfg):
    """(mux, demux, margin) insertion terms in dB."""
    cl = cfg.get("classical_link", {})
    return (float(cl.get("mux_insertion_db", 0.0)),
            float(cl.get("demux_insertion_db", 0.0)),
            float(cl.get("system_margin_db", 0.0)))
