"""Frozen classical allocation, quantum candidate domain, and the component
ledger (Methods: classical allocation and reference planes)."""
from dataclasses import dataclass, field
import numpy as np
from units import nm_to_hz, hz_to_nm, dbm_to_w

# The 35 active carrier frequencies of the fixed classical allocation.
# This is a GENERATED scenario, not a measured deployed allocation.
LEGACY_CLASSICAL_GHZ = [191600, 191700, 191800, 191900, 192700, 192800, 192900,
                        193000, 193100, 193300, 193400, 193500, 193600, 193700,
                        193800, 193900, 194000, 194100, 194200, 194300, 194400,
                        194500, 194600, 194700, 194800, 194900, 195100, 195200,
                        195300, 195400, 195500, 195600, 195700, 195800, 195900]


@dataclass
class Deployment:
    """A frozen classical deployment.  Frequencies never change when the quantum
    search band, length, or power changes."""
    freqs_hz: np.ndarray
    launch_w: np.ndarray
    direction: str = "A_to_B"
    quantum_directions: tuple = ("A_to_B", "B_to_A", "A_to_B")
    # transmitter terminal of the classical comb
    classical_tx_terminal: str = "A"
    quantum_rx_terminal: tuple = ("B", "A", "B")

    @property
    def n_channels(self):
        return len(self.freqs_hz)

    def copropagating(self, pass_index):
        """True when the quantum traversal co-propagates with the classical comb."""
        return self.quantum_directions[pass_index] == self.direction

    def local_transmitter(self, pass_index):
        """True when the classical transmitters sit at the terminal that hosts
        this pass's quantum receiver (direct, non-span leakage path)."""
        return self.quantum_rx_terminal[pass_index] == self.classical_tx_terminal

    def to_rows(self):
        rows = []
        for i, (f, p) in enumerate(zip(self.freqs_hz, self.launch_w)):
            rows.append({"channel_index": i,
                         "frequency_hz": float(f),
                         "frequency_ghz": float(f / 1e9),
                         "wavelength_nm": float(hz_to_nm(f)),
                         "launch_power_w": float(p),
                         "launch_power_dbm": float(10 * np.log10(p / 1e-3)),
                         "tx_terminal": self.classical_tx_terminal,
                         "direction": self.direction})
        return rows


def legacy_deployment(launch_dbm=-5.0):
    f = np.array(LEGACY_CLASSICAL_GHZ, dtype=float) * 1e9
    p = np.full(f.shape, dbm_to_w(launch_dbm))
    return Deployment(freqs_hz=f, launch_w=p)


# ------------------------------------------------------------- candidate domain
def candidate_grid(f_lo_hz, f_hi_hz, df_hz, extra_points=()):
    """Uniform frequency grid plus explicitly inserted feature locations."""
    n = int(np.floor((f_hi_hz - f_lo_hz) / df_hz)) + 1
    g = f_lo_hz + df_hz * np.arange(n)
    g = np.concatenate([g, np.asarray(list(extra_points), dtype=float)])
    g = np.unique(np.clip(g, f_lo_hz, f_hi_hz))
    return g


def guard_boundaries(dep, guard_hz):
    """Frequencies at exactly +/- guard_hz from every carrier: the inclusive
    boundary points of the guard-band exclusion under a >= convention."""
    return np.sort(np.concatenate([dep.freqs_hz - guard_hz, dep.freqs_hz + guard_hz]))


def domain_masks(f_grid, dep, guard_hz, hw_lo_hz, hw_hi_hz,
                 reserved_window_hz=None, restrict_to_reserved=False):
    """Return (hardware_allowed, guard_allowed, search_allowed, reason).

    Guard rule uses `>=`: a candidate exactly guard_hz from a carrier is allowed.
    """
    f_grid = np.asarray(f_grid, dtype=float)
    hw = (f_grid >= hw_lo_hz) & (f_grid <= hw_hi_hz)
    d = np.abs(f_grid[:, None] - dep.freqs_hz[None, :])
    guard = np.min(d, axis=1) >= guard_hz - 1e-6      # 1 uHz tolerance
    if restrict_to_reserved and reserved_window_hz is not None:
        lo, hi = min(reserved_window_hz), max(reserved_window_hz)
        search = (f_grid >= lo) & (f_grid <= hi)
    else:
        search = np.ones_like(hw, dtype=bool)
    allowed = hw & guard & search
    reason = np.array(["allowed"] * len(f_grid), dtype=object)
    reason[~search] = "outside_reserved_search_window"
    reason[~guard] = "guard_band"
    reason[~hw] = "outside_hardware_domain"
    return hw, guard, search, allowed, reason


# --------------------------------------------------------------- component ledger
@dataclass
class ComponentLedger:
    """Per-pass transmission ledger with explicit reference planes.

    eta_tx : launch-side (quantum transmitter -> fibre input) power transmission
    eta_rx : receive-side lumped, spectrally flat power transmission
             (fibre output -> filter input)
    The wavelength-selective filter T_q(f) is applied SEPARATELY and exactly once:
    to the signal as the mode-averaged value, and to in-span background inside the
    occupancy integral.
    """
    eta_tx: float
    eta_rx: float
    note: str = ""

    @property
    def eta_lumped(self):
        return self.eta_tx * self.eta_rx

    def rows(self, pass_index, length_m):
        return [
            {"pass": pass_index + 1, "path": "quantum_signal_launch",
             "plane_in": "quantum source", "plane_out": "fibre input",
             "transmission": self.eta_tx, "spectral": "flat",
             "included_once": True, "note": self.note},
            {"pass": pass_index + 1, "path": "quantum_signal_span",
             "plane_in": "fibre input", "plane_out": "fibre output",
             "transmission": "exp(-a(lambda_q) L)", "spectral": "yes",
             "included_once": True, "note": f"L = {length_m:.1f} m"},
            {"pass": pass_index + 1, "path": "quantum_signal_receive",
             "plane_in": "fibre output", "plane_out": "filter input",
             "transmission": self.eta_rx, "spectral": "flat",
             "included_once": True, "note": self.note},
            {"pass": pass_index + 1, "path": "quantum_signal_filter",
             "plane_in": "filter input", "plane_out": "accepted mode",
             "transmission": "int |psi|^2 T_q(f) df", "spectral": "yes",
             "included_once": True, "note": "applied once; noise sees T_q(f) in the nu integral"},
            {"pass": pass_index + 1, "path": "in_span_background",
             "plane_in": "distributed in fibre", "plane_out": "accepted mode",
             "transmission": "eta_rx * T_q(f)", "spectral": "yes",
             "included_once": True,
             "note": "in-span propagation already inside the Raman/FWM integrals"},
            {"pass": pass_index + 1, "path": "classical_leakage",
             "plane_in": "classical transmitter", "plane_out": "accepted mode",
             "transmission": "T_sys(f) (path-specific, includes isolation)",
             "spectral": "yes", "included_once": True,
             "note": "local vs remote path selected per pass"},
        ]
