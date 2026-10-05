"""Fibre attenuation model with explicit provenance (Methods: fibre attenuation).

No measured attenuation spectrum and no manufacturer datasheet were supplied to
this run, so the baseline curve is an ILLUSTRATIVE ASSUMPTION built from the
standard three-term silica functional form and anchored to one declared value at
1550 nm.  It is not a measured spectrum and not a proven pointwise bound.
"""
import numpy as np
from units import db_per_km_to_per_m


class AttenuationModel:
    """alpha(lambda) [dB/km] = A/lambda^4 + B exp(-C/lambda) + D, lambda in um."""

    def __init__(self, rayleigh_a=0.80, ir_b=7.81e11, ir_c=48.48,
                 anchor_nm=1550.0, anchor_db_km=0.190,
                 valid_nm=(1450.0, 1650.0),
                 provenance="illustrative_assumption"):
        self.A = float(rayleigh_a)
        self.B = float(ir_b)
        self.C = float(ir_c)
        self.valid_nm = tuple(valid_nm)
        self.provenance = provenance
        lam = anchor_nm * 1e-3
        self.D = float(anchor_db_km) - (self.A / lam ** 4 + self.B * np.exp(-self.C / lam))
        self.anchor_nm = float(anchor_nm)
        self.anchor_db_km = float(anchor_db_km)

    def db_per_km(self, lam_nm, allow_extrapolation=False):
        lam_nm = np.asarray(lam_nm, dtype=float)
        lo, hi = self.valid_nm
        if not allow_extrapolation and (np.any(lam_nm < lo) or np.any(lam_nm > hi)):
            raise ValueError(f"wavelength outside the model's declared validity {self.valid_nm} nm")
        lam = lam_nm * 1e-3
        return self.A / lam ** 4 + self.B * np.exp(-self.C / lam) + self.D

    def a_per_m(self, lam_nm, allow_extrapolation=False):
        return db_per_km_to_per_m(self.db_per_km(lam_nm, allow_extrapolation))

    def rows(self, lam_nm_list):
        return [{"wavelength_nm": float(l),
                 "attenuation_db_km": float(self.db_per_km(l)),
                 "provenance_category": self.provenance,
                 "basis": (f"three-term silica form (Rayleigh + IR + constant), "
                           f"anchored to {self.anchor_db_km} dB/km at "
                           f"{self.anchor_nm} nm; coefficients not fitted to "
                           f"supplied measurements")}
                for l in lam_nm_list]


class Dispersion:
    """Group-velocity dispersion from D and S at a reference wavelength."""

    def __init__(self, D_ps_nm_km=17.0, S_ps_nm2_km=0.058, lam0_nm=1550.0,
                 n_group=1.4682, beta2_s2_m=None, beta3_s3_m=None):
        from units import C_LIGHT
        self.D = D_ps_nm_km * 1e-12 / (1e-9 * 1e3)      # s/(m*m) = s/m^2
        self.S = S_ps_nm2_km * 1e-12 / (1e-18 * 1e3)    # s/m^3
        self.lam0 = lam0_nm * 1e-9
        self.n_group = float(n_group)
        c = C_LIGHT
        self.beta2 = (-self.D * self.lam0 ** 2 / (2 * np.pi * c)
                      if beta2_s2_m is None else float(beta2_s2_m))
        self.beta3 = ((self.lam0 / (2 * np.pi * c)) ** 2 *
                      (self.lam0 ** 2 * self.S + 2 * self.lam0 * self.D)
                      if beta3_s3_m is None else float(beta3_s3_m))
        self.f0 = c / self.lam0
        self.beta1 = self.n_group / c

    def beta(self, f_hz):
        """beta(f) retaining constant, linear, beta2 and beta3 terms."""
        from units import C_LIGHT
        w = 2 * np.pi * (np.asarray(f_hz, dtype=float) - self.f0)
        beta0 = 2 * np.pi * self.f0 * self.n_group / C_LIGHT
        return (beta0 + self.beta1 * w + 0.5 * self.beta2 * w ** 2
                + self.beta3 * w ** 3 / 6.0)

    def walkoff_co(self, f1_hz, f2_hz):
        """|d| = |1/vg(f1) - 1/vg(f2)| in s/m for co-propagating waves."""
        w1 = 2 * np.pi * (np.asarray(f1_hz, float) - self.f0)
        w2 = 2 * np.pi * (np.asarray(f2_hz, float) - self.f0)
        d1 = self.beta1 + self.beta2 * w1 + 0.5 * self.beta3 * w1 ** 2
        d2 = self.beta1 + self.beta2 * w2 + 0.5 * self.beta3 * w2 ** 2
        return np.abs(d1 - d2)

    def walkoff_counter(self, f1_hz, f2_hz):
        """|d| = 1/vg(f1) + 1/vg(f2) for counter-propagating waves."""
        w1 = 2 * np.pi * (np.asarray(f1_hz, float) - self.f0)
        w2 = 2 * np.pi * (np.asarray(f2_hz, float) - self.f0)
        d1 = self.beta1 + self.beta2 * w1 + 0.5 * self.beta3 * w1 ** 2
        d2 = self.beta1 + self.beta2 * w2 + 0.5 * self.beta3 * w2 ** 2
        return np.abs(d1 + d2)


# --------------------------------------------------------------- tabulated data
_SMF28_ROW_PROVENANCE = {
    # Datasheet MAXIMUM specifications quoted in the file header (bounds at the
    # stated wavelengths and conditions, not a measured spectral curve).
    1310.0: "manufacturer_specification", 1383.0: "manufacturer_specification",
    1490.0: "manufacturer_specification", 1550.0: "manufacturer_specification",
    1625.0: "manufacturer_specification",
    # Points between datasheet anchors, supplied by the file author.
    1520.0: "derived", 1565.0: "derived", 1580.0: "derived",
    # Outside the datasheet's 1310-1625 nm scope.
    1290.0: "illustrative_assumption", 1300.0: "illustrative_assumption",
    1650.0: "illustrative_assumption", 1670.0: "illustrative_assumption",
    1700.0: "illustrative_assumption",
}


class TabulatedAttenuation:
    """alpha(lambda) from a two-column CSV, with PER-ROW provenance.

    The supplied SMF-28 file mixes datasheet MAXIMUM specifications, points the
    file author placed between anchors, and points outside the datasheet's
    stated scope.  Interpolating those rows defines an ASSUMED curve; it is
    neither a measured spectrum nor a proven pointwise bound at intervening
    wavelengths.  `spec_range_nm` records where datasheet anchors exist at all.
    """

    def __init__(self, csv_path, valid_nm=None):
        import csv as _csv
        lam, alpha = [], []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in _csv.reader(fh):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                try:
                    lam.append(float(row[0])); alpha.append(float(row[1]))
                except ValueError:
                    continue                       # header line
        o = np.argsort(lam)
        self.lam_nm = np.asarray(lam, float)[o]
        self.alpha_db_km_tab = np.asarray(alpha, float)[o]
        self.csv_path = csv_path
        self.valid_nm = tuple(valid_nm) if valid_nm else (float(self.lam_nm[0]),
                                                          float(self.lam_nm[-1]))
        specs = [l for l in self.lam_nm
                 if _SMF28_ROW_PROVENANCE.get(float(l)) == "manufacturer_specification"]
        self.spec_range_nm = (min(specs), max(specs)) if specs else None
        self.provenance = "mixed_per_row"
        self.interpolation = "linear in wavelength"

    def db_per_km(self, lam_nm, allow_extrapolation=False):
        lam_nm = np.asarray(lam_nm, dtype=float)
        lo, hi = self.valid_nm
        if not allow_extrapolation and (np.any(lam_nm < lo) or np.any(lam_nm > hi)):
            raise ValueError(f"wavelength outside the tabulated range {self.valid_nm} nm")
        return np.interp(lam_nm, self.lam_nm, self.alpha_db_km_tab)

    def a_per_m(self, lam_nm, allow_extrapolation=False):
        return db_per_km_to_per_m(self.db_per_km(lam_nm, allow_extrapolation))

    def rows(self, lam_nm_list=None):
        out = []
        for l, a in zip(self.lam_nm, self.alpha_db_km_tab):
            cat = _SMF28_ROW_PROVENANCE.get(float(l), "illustrative_assumption")
            out.append({"wavelength_nm": float(l), "attenuation_db_km": float(a),
                        "provenance_category": cat,
                        "basis": {"manufacturer_specification":
                                  "Corning SMF-28 Ultra PI-1424-AEN datasheet MAXIMUM "
                                  "at this wavelength; a bound, not a measurement",
                                  "derived":
                                  "point placed between datasheet anchors by the file "
                                  "author; an assumed interpolation, not a spec",
                                  "illustrative_assumption":
                                  "outside the datasheet's 1310-1625 nm scope; "
                                  "extrapolation, must not be quoted as spec"}[cat]})
        return out
