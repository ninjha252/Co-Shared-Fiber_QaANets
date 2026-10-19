#!/usr/bin/env python
"""The four main figures plus the supplementary diagnostics figure
(Methods: figures).

No science is recomputed here; every panel reads a CSV written by a run script.
Exports vector PDF and SVG plus a 400 dpi PNG preview, with embedded fonts.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

MAIN_POINT = "op_5km_m10dBm"          # 5 km per traversal, -10 dBm/ch, 1 GHz margin
SECOND_POINT = "op_3km_m13dBm"        # predeclared interior example

# --------------------------------------------------------------- figure style
# Three variants are produced from the same code:
#   "default"   captioned figures for the working record (sans, compact)
#   "nocaption" manuscript figures: NO figure title drawn on the canvas,
#               Times New Roman throughout, larger axis and tick fonts
#   "final"     the manuscript set in the nocaption style, written to its own
#               directory together with the supplementary figures
BASE_RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "figure.dpi": 140,
    "savefig.bbox": "tight", "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
    "axes.prop_cycle": plt.cycler(color=["#1b4079", "#c1440e", "#2e7d32",
                                         "#6a1b9a", "#00695c", "#b71c1c"]),
}
SCALE = 1.0                 # multiplies every explicit fontsize and marker size
FIGSCALE = 1.0              # multiplies the figure canvas
OUTDIR = FIG
SUPTITLE = True


def fs(x):
    """Explicit font size, scaled with the active style."""
    return x * SCALE


def ms(x):
    """Explicit marker size, scaled with the active style."""
    return x * (1.0 + 0.55 * (SCALE - 1.0))


def size(w, h):
    """Figure size, scaled with the active style."""
    return (w * FIGSCALE, h * FIGSCALE)


def set_style(variant="default"):
    """Activate one of the two output variants."""
    global SCALE, FIGSCALE, OUTDIR, SUPTITLE
    rc = dict(BASE_RC)
    if variant in ("nocaption", "final"):
        SCALE, FIGSCALE, SUPTITLE = 1.55, 1.18, False
        OUTDIR = os.path.join(FIG, variant)
        rc.update({"font.family": "serif",
                   "font.serif": ["Times New Roman", "Times", "STIXGeneral",
                                  "DejaVu Serif"],
                   "mathtext.fontset": "stix",
                   "font.size": 12.5, "axes.labelsize": 13,
                   "axes.titlesize": 13, "xtick.labelsize": 12,
                   "ytick.labelsize": 12, "legend.fontsize": 10.5,
                   "axes.linewidth": 0.9, "lines.linewidth": 1.5,
                   "xtick.major.width": 0.9, "ytick.major.width": 0.9,
                   "xtick.major.size": 4.0, "ytick.major.size": 4.0})
    else:
        SCALE, FIGSCALE, SUPTITLE = 1.0, 1.0, True
        OUTDIR = FIG
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams.update(rc)


def suptitle(fig, text, **kw):
    """Draw the figure title only in the captioned variant."""
    if not SUPTITLE:
        return None
    return fig.suptitle(text, **kw)


set_style("default")
DASH = ["-", "--", "-.", ":"]
MARK = ["o", "s", "^", "D", "v", "P"]


def save(fig, name, extra=None):
    extra = [a for a in (extra or []) if a is not None] or None
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"),
                    dpi=(400 if ext == "png" else None),
                    bbox_extra_artists=extra)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.svg/.png")


def rd(run_id, fname):
    """Read a stored result table.

    Large per-candidate scans are distributed gzip-compressed to keep the
    repository small; pandas reads either form transparently.
    """
    p = os.path.join(RES, run_id, fname)
    if os.path.exists(p):
        return pd.read_csv(p)
    if os.path.exists(p + ".gz"):
        return pd.read_csv(p + ".gz")
    return None


# line style of each receiver's link-budget line, used in Figs. 4 and S2
LB_STYLE = {"rx_1g25_measured": "--", "rx_10g_apd": ":"}
# colour and marker of each receiver in Fig. S2 (b, c)
LB_COLOR = {"rx_1g25_measured": ("#e07000", "o"), "rx_10g_apd": ("#228b22", "s")}


def link_budget_lines(L):
    """(length, minimum launch power, receiver, line style) for every declared
    receiver.  The attenuation and the insertion terms are read from the
    stored link-budget run, so the line is the one the run evaluated."""
    lbp = rd("illustrative_baseline", "link_budget_points.csv")
    if lbp is None:
        return []
    import config as _cf, link_budget as _lb
    rxs = {r["id"]: r for r in _lb.receivers(_cf.load("illustrative_baseline.yaml"))}
    rows = lbp[lbp["point_kind"] != "predeclared"]
    out = []
    Lc = np.linspace(np.min(L), np.max(L), 200)
    for rid, g in rows.groupby("receiver", sort=False):
        r0 = g.iloc[0]
        Pmin = _lb.minimum_launch_dbm(Lc, r0["rx_sensitivity_dbm"],
                                      r0["worst_classical_alpha_db_km"],
                                      r0["demux_db"], r0["margin_db"])
        out.append((Lc, Pmin, rxs.get(rid, {"id": rid, "data_rate_gbps": np.nan}),
                    LB_STYLE.get(rid, "-.")))
    return out


# ----------------------------------------------------------------- Figure 1
def figure1():
    v = rd("validation", "optical_validation.csv")
    c = rd("validation", "optical_convergence.csv")
    fig, ax = plt.subplots(1, 3, figsize=size(7.4, 2.75), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.06)
    show = [0.05, 0.2, 0.9]
    for i, T in enumerate(show):
        sub = v[(v["T"] == T) & (v["nu"] > 0)].sort_values("nu")
        ax[0].plot(sub["nu"], sub["s_analytic"], DASH[i % 4], color=f"C{i}")
        ax[0].plot(sub["nu"], sub["s_numeric"], MARK[i % 6], color=f"C{i}",
                   ms=ms(3.4), mfc="none", mew=0.8, ls="none")
        ax[1].plot(sub["nu"], sub["r_analytic"], DASH[i % 4], color=f"C{i}")
        ax[1].plot(sub["nu"], sub["r_numeric"], MARK[i % 6], color=f"C{i}",
                   ms=ms(3.4), mfc="none", mew=0.8, ls="none")
    for a, lab in ((ax[0], r"acceptance $s$"), (ax[1], r"conditional mixing $r$")):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel(r"final occupancy $\nu$ per polarization")
        a.set_ylabel(lab)
    ax[0].set_ylim(3e-2, 1.4)
    h = [Line2D([], [], color=f"C{i}", ls=DASH[i % 4], marker=MARK[i % 6],
                mfc="none", ms=ms(3.4), label=f"$T={T}$") for i, T in enumerate(show)]
    hs = [Line2D([], [], color="0.3", ls="-", label="closed-form weights"),
          Line2D([], [], color="0.3", ls="none", marker="o", mfc="none", ms=ms(3.4),
                 label="independent dilation")]
    ax[0].legend(handles=h, ncol=3, columnspacing=0.9, handlelength=2.0,
                 loc="upper left", bbox_to_anchor=(0.0, -0.25), fontsize=fs(6.2))
    ax[1].legend(handles=hs, ncol=1, handlelength=2.0, loc="upper left",
                 bbox_to_anchor=(0.0, -0.25), fontsize=fs(6.2))
    # Panel (c) carries no annotation: each convergence series has its own
    # colour AND its own marker shape, and the production setting of each
    # series is overplotted with a red star.  The caption explains both.
    cc_ = ["#e07000", "#daa520", "#228b22", "#8b4513"]   # dark orange,
    #                                goldenrod, forest green, brown
    mk_ = ["o", "s", "^", "D"]
    PROD = "#b3122a"          # production cutoff (tail tolerance 1e-16)
    TARGET = "#1a1a1a"        # target tolerance 1e-9 (kept neutral so it does
    #                           not compete with the warm series colours)
    for k, T in enumerate(sorted(c["T"].unique())):
        sub = c[c["T"] == T].sort_values("two_env_tail_bound")
        err = np.maximum(sub["choi_trace_norm_error"], 1e-18)
        ax[2].loglog(sub["two_env_tail_bound"], err, "-", marker=mk_[k % 4],
                     ms=ms(3.6), mfc="none", mew=1.0, color=cc_[k % 4], lw=0.9,
                     label=fr"$T={T}$, $\nu={sub['nu'].iloc[0]:g}$")
        # the tightest cutoff of each series is the setting used in production
        ax[2].loglog(sub["two_env_tail_bound"].iloc[0], err.iloc[0], "*",
                     ms=ms(8.0), color=PROD, mec=PROD, mfc="none", mew=1.3,
                     zorder=5)
    ax[2].axhline(1e-9, color=TARGET, lw=1.1, ls="--")
    h2 = [Line2D([], [], color=TARGET, ls="--", lw=1.1, label="target tolerance"),
          Line2D([], [], color=PROD, marker="*", ls="none", ms=ms(8.0),
                 mfc="none", mew=1.3, label="production cutoff")]
    ax[2].set_xlim(1e-18, 1e-7)
    ax[2].set_ylim(1e-18, 1e-6)
    ax[2].set_xlabel("discarded thermal probability")
    ax[2].set_ylabel(r"$\|J_{\rm num}-J_{\rm formula}\|_1$ (trace norm)")
    hc, lc = ax[2].get_legend_handles_labels()
    ax[2].legend(hc + h2, lc + [x.get_label() for x in h2],
                 loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=2,
                 fontsize=fs(6.2), columnspacing=0.9)
    for a in ax:
        a.get_legend().set_in_layout(False)
    for a, t in zip(ax, ("(a) acceptance", "(b) conditional mixing",
                         "(c) numerical discrepancy")):
        a.set_title(t, loc="left")
    suptitle(fig, "Figure 1 | Optical-channel verification", x=0.005, ha="left",
                 fontsize=fs(9))
    save(fig, "figure1_optical_channel_verification",
         extra=[a.get_legend() for a in ax] + [fig._suptitle if SUPTITLE else None] +
               [a.title for a in ax])


# ----------------------------------------------------------------- Figure 2
def figure2():
    d = rd("validation", "decoder_validation.csv")
    k = rd("validation", "key_sensitivity.csv")
    PNAME = {"baseline": "baseline", "xpm_dominant": "strong XPM"}
    fig, ax = plt.subplots(1, 3, figsize=size(7.4, 2.75), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.10)
    base = d[d["noise_path"] == "baseline"]
    dm = base[base["method"] == "density_matrix"]
    idq = dm[dm["key_case"] == "identity"].sort_values("q")
    ncq = dm[dm["key_case"] == "fixed_nonclifford_ry"].sort_values("q")
    pe = base[base["method"] == "pauli_enumeration"].sort_values("q")
    m = idq["q"] > 0
    ax[0].loglog(idq["q"][m], 1 - idq["C_bound"][m], "-", color="0.25", lw=1.4,
                 label=r"$1-C_{7,1}(q)$ (bound on the infidelity)")
    ax[0].loglog(idq["q"][m], 1 - idq["Fe_exact"][m], "-", color="C0",
                 label="identity keys (density matrix)")
    ax[0].loglog(ncq["q"][ncq["q"] > 0], 1 - ncq["Fe_exact"][ncq["q"] > 0], "--",
                 color="C1", label="non-Clifford keys (density matrix)")
    pm = pe["q"] > 0
    ax[0].loglog(pe["q"][pm][::3], 1 - pe["Fe_exact"][pm][::3], "o", color="C0",
                 ms=ms(3.5), mfc="none", ls="none",
                 label="identity keys (Pauli enumeration)")
    b = idq.iloc[(idq["u"] - 1.0).abs().argsort()[:1]]
    ax[0].plot(b["q"], 1 - b["Fe_exact"], "*", color="C5", ms=ms(9), ls="none",
               label="reference noise point")
    ax[0].set_xlabel(r"affected-location probability $q$")
    ax[0].set_ylabel(r"logical infidelity $1-F_e$")
    ax[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=1,
                 fontsize=fs(6.0), handlelength=2.2)
    ax[0].set_title("(a) exact recovery vs the bound", loc="left",
                    fontsize=fs(7.8))

    gaps = {}
    for pth, cc, ls in (("baseline", "C0", "-"), ("xpm_dominant", "C3", "-.")):
        g = d[(d["noise_path"] == pth) & (d["method"] == "density_matrix")]
        gi = g[g["key_case"] == "identity"].sort_values("q")
        gn = g[g["key_case"] == "fixed_nonclifford_ry"].sort_values("q")
        gi, gn = gi[gi["q"] > 0], gn[gn["q"] > 0]
        gaps[pth] = (gi, gn, cc, ls)
        ax[1].semilogx(gi["q"], gi["Fe_exact"] - gi["C_bound"], ls, color=cc,
                       label=f"{PNAME[pth]}, identity")
        ax[1].semilogx(gn["q"], gn["Fe_exact"] - gn["C_bound"], ":", color=cc,
                       lw=1.4, label=f"{PNAME[pth]}, non-Clifford")
    ax[1].axhline(0, color="0.5", lw=0.8, ls=":")
    ax[1].set_xlabel(r"affected-location probability $q$")
    ax[1].set_ylabel(r"$F_e^{\rm exact}-C_{7,1}(q)$")
    ax[1].set_title("(b) bound conservatism", loc="left", fontsize=fs(7.8))
    ins = ax[1].inset_axes([0.10, 0.50, 0.42, 0.40])
    gi, gn, cc, ls = gaps["baseline"]
    ins.loglog(gi["q"], (gi["Fe_exact"] - gi["C_bound"]).clip(lower=1e-18), ls,
               color=cc, lw=1.0)
    ins.loglog(gn["q"], (gn["Fe_exact"] - gn["C_bound"]).clip(lower=1e-18), ":",
               color=cc, lw=1.3)
    ins.set_ylim(1e-12, 1e-1)
    ins.set_title("baseline only (log)", fontsize=fs(5.6), pad=2)
    ins.tick_params(labelsize=fs(5), length=2, pad=1)
    ins.minorticks_off()
    ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=1,
                 fontsize=fs(6.0), handlelength=2.2)

    FLOOR = float(d.loc[d["method"] == "pauli_vs_dm_residual", "Fe_exact"].max())
    for pth, cc, mk in (("baseline", "C0", "s"), ("xpm_dominant", "C3", "^")):
        rr = k[(k["key_set_id"] == "random_range") &
               (k["noise_path"] == pth)].dropna(subset=["Fe_min"])
        rr = rr[rr["q"] > 0]
        sp = (rr["Fe_max"] - rr["Fe_min"]).clip(lower=1e-17)
        ax[2].loglog(rr["q"], sp, mk + "-", color=cc, ms=ms(3.5),
                     label=f"{PNAME[pth]}: 16 random key sets")
        g = d[(d["noise_path"] == pth) & (d["method"] == "density_matrix")]
        gi = g[g["key_case"] == "identity"].sort_values("q")
        gn = g[g["key_case"] == "fixed_nonclifford_ry"].sort_values("q")
        gi, gn = gi[gi["q"] > 0], gn[gn["q"] > 0]
        fx = np.abs(gn["Fe_exact"].to_numpy() - gi["Fe_exact"].to_numpy()).clip(1e-17)
        ax[2].loglog(gi["q"], fx, "--", color=cc, lw=0.9,
                     label=f"{PNAME[pth]}: fixed vs identity")
    ax[2].axhspan(1e-17, max(FLOOR, 1e-17), color="0.88", zorder=0)
    ax[2].text(0.97, 1e-15, "measured comparison residual\n(numerical reference)",
               fontsize=fs(5.6), color="0.35", va="center", ha="right",
               transform=ax[2].get_yaxis_transform())
    ax[2].set_xlabel(r"affected-location probability $q$")
    ax[2].set_ylabel(r"key dependence of $F_e^{\rm exact}$")
    ax[2].legend(loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=1,
                 fontsize=fs(6.0), handlelength=2.2)
    ax[2].set_ylim(1e-17, 1e0)
    ax[2].set_title("(c) sampled key dependence", loc="left", fontsize=fs(7.8))
    legs = [a.get_legend() for a in ax]
    for lg in legs:
        lg.set_in_layout(False)
    suptitle(fig, "Figure 2 | Logical fidelity and the correction bound",
                 x=0.005, ha="left", fontsize=fs(9))
    save(fig, "figure2_logical_fidelity_and_bound",
         extra=legs + [fig._suptitle if SUPTITLE else None] +
               [a.title for a in ax])


# ----------------------------------------------------------------- Figure 3
SEL_STYLE = {
    "max_Ylb_subject_to_Fmin": ("max $Y_{\\rm lb}$ s.t. $F_{\\min}$", "C2", "v"),
    "max_T_over_Omega_q": ("max $T$ (loss only)", "C1", "^"),
    "min_q_over_Omega_F": ("min $q$ (quality)", "C4", "D"),
}


def _selector_marks(sel):
    """(label, colour, marker, wavelength) for the comparators that are defined."""
    out = []
    for rule, (lab, col, mk) in SEL_STYLE.items():
        row = sel[sel["selection_rule"] == rule]
        if len(row) and not pd.isna(row["lambda_selected_nm"].iloc[0]):
            out.append((lab, col, mk, float(row["lambda_selected_nm"].iloc[0])))
    ref = sel[sel["selection_rule"].astype(str).str.startswith("fixed reference")]
    if len(ref):
        lam = ref["lambda_selected_nm"].iloc[0]
        if not pd.isna(lam):
            out.append(("fixed reference", "C5", "P", float(lam)))
    return out


def figure3():
    w = rd(MAIN_POINT, "wavelength_scan.csv")
    sel = rd(MAIN_POINT, "selection_summary.csv")
    chan = rd(MAIN_POINT, "classical_channels.csv")
    reg = rd(MAIN_POINT, "certified_regions.csv")
    if w is None:
        print("  figure3 skipped: no scan for", MAIN_POINT)
        return
    al = (w["exclusion_reason"] == "allowed").to_numpy()
    lam = w["wavelength_nm"].to_numpy()[al]
    marks = _selector_marks(sel)

    fig, A = plt.subplots(4, 1, figsize=size(7.0, 7.6), sharex=True,
                          layout="constrained")
    fig.get_layout_engine().set(h_pad=0.04, hspace=0.05)
    for a in A:
        for f in chan["wavelength_nm"]:
            a.axvline(f, color="0.92", lw=0.5, zorder=0)
    # (a) propagated occupancy contributions
    A[0].semilogy(lam, w["nu_final"].to_numpy()[al], "o", ms=ms(3.0), color="0.78",
                  mec="none", label=r"$\nu$ total")
    for nm_, lab, cc in (("nu_R_final", r"$\nu_R$ Raman", "C0"),
                         ("nu_FWM_final", r"$\nu_{\rm FWM}$", "C1"),
                         ("nu_leak_final", r"$\nu_{\rm leak}$", "C2")):
        A[0].semilogy(lam, np.maximum(w[nm_].to_numpy()[al], 1e-16), ".", ms=ms(2.0),
                      color=cc, label=lab)
    A[0].set_ylabel(r"final $\nu$ per polarization")
    A[0].set_ylim(1e-16, 3e0)
    A[0].legend(ncol=4, loc="upper center", markerscale=3, handletextpad=0.2,
                columnspacing=0.9, bbox_to_anchor=(0.5, 1.02), fontsize=fs(6.2))
    A[0].text(0.012, 0.06, "FWM components are delta lines (narrow-line limit)",
              transform=A[0].transAxes, fontsize=fs(5.8), color="C1", va="bottom")
    # (b) acceptance
    A[1].plot(lam, w["s"].to_numpy()[al], ".", ms=ms(2.4), color="C4")
    A[1].set_ylabel(r"acceptance $s$")
    # (c) certificate and exact fidelity
    A[2].plot(lam, w["C_bound"].to_numpy()[al], ".", ms=ms(2.4), color="C3",
              label=r"$C_{7,1}(q)$ (lower bound, key independent)")
    ev = w[(w["exact_decoder_evaluated"] == True) & al]
    A[2].plot(ev["wavelength_nm"], ev["Fe_exact"], ".", ms=ms(1.8), color="C0",
              label=r"$F_e$ exact, identity keys (Pauli enumeration)")
    F_MIN = 0.99
    A[2].axhline(F_MIN, color="0.4", ls=":", lw=0.9)
    cv = w["C_bound"].to_numpy()[al]
    fv = ev["Fe_exact"].to_numpy()
    lo = min(np.nanmin(cv), np.nanmin(fv)) if len(fv) else np.nanmin(cv)
    hi = max(np.nanmax(cv), np.nanmax(fv)) if len(fv) else np.nanmax(cv)
    lo = min(lo, F_MIN)              # keep the threshold in view
    A[2].set_ylim(lo - 0.12 * (hi - lo), hi + 0.10 * (hi - lo))
    A[2].text(0.012, F_MIN, r"$F_{\min}=0.99$", transform=A[2].get_yaxis_transform(),
              fontsize=fs(6), color="0.35", va="bottom")
    A[2].set_ylabel("conditional fidelity")
    # (d) yields
    A[3].semilogy(lam, w["Y_ub"].to_numpy()[al], ".", ms=ms(1.8), color="0.6",
                  label=r"$Y_{\rm ub}=s^7$")
    # colours follow panel (c): Y_lb pairs with C, the exact Y with F_e
    A[3].semilogy(lam, w["Y_lb"].to_numpy()[al], ".", ms=ms(2.4), color="C3",
                  label=r"$Y_{\rm lb}=s^7C_{7,1}$")
    A[3].semilogy(ev["wavelength_nm"], ev["Y_exact"], ".", ms=ms(1.8), color="C0",
                  label=r"$Y=s^7F_e$ (identity keys)")
    A[3].axhline(1e-6, color="0.4", ls=":", lw=0.9)
    A[3].text(0.012, 0.06, r"$Y_{\min}=10^{-6}$", transform=A[3].transAxes,
              fontsize=fs(6), color="0.35")
    A[3].set_ylabel("fidelity yield per block")
    A[3].set_xlabel("quantum wavelength (nm)")
    # certified components: stems for zero-width, bands only where finite
    if reg is not None:
        wq = reg[(reg["region"] == "W_q") & (reg["n_intervals"] > 0)]
        n_iso = n_band = 0
        for _, rr in wq.iterrows():
            width = float(rr["lambda_hi_nm"]) - float(rr["lambda_lo_nm"])
            if width < 1e-9:
                A[3].axvline(float(rr["lambda_lo_nm"]), color="C2", lw=0.5,
                             alpha=0.5, zorder=0)
                n_iso += 1
            else:
                A[3].axvspan(rr["lambda_lo_nm"], rr["lambda_hi_nm"], color="C2",
                             alpha=0.22, lw=0, zorder=0)
                n_band += 1
    # comparator selections: thin guide lines at identical wavelengths in every
    # panel (coincident selections share one line and one legend entry)
    ls_cycle = ["-", "--", ":", "-."]
    seen = {}
    for lab, col, mk, lm in marks:
        key = round(lm, 6)
        seen.setdefault(key, []).append((lab, col))
    hmark = []
    for j, (lm, labs) in enumerate(sorted(seen.items())):
        col = labs[0][1]
        for a in A:
            a.axvline(lm, color=col, lw=0.9, ls=ls_cycle[j % 4], alpha=0.85,
                      zorder=1)
        hmark.append(Line2D([], [], color=col, lw=1.0, ls=ls_cycle[j % 4],
                            label=" = ".join(l for l, _ in labs) + f": {lm:.3f} nm"))
    # one legend below the panels: the curves of (c) and (d), then the
    # selector guide lines, so no legend sits on the data
    hc, lc = A[2].get_legend_handles_labels()
    hd, ld = A[3].get_legend_handles_labels()
    fig.legend(handles=hc + hd + hmark, labels=lc + ld + [h.get_label() for h in hmark],
               loc="outside lower center", ncol=2, markerscale=3,
               fontsize=fs(6.2), columnspacing=1.4)
    for a, tg in zip(A, "abcd"):
        a.set_title(f"({tg})", loc="left", fontsize=fs(7.5))
    suptitle(fig, "Figure 3 | Wavelength selection at 5 km per traversal, "
                 "$-10$ dBm/channel, 50 GHz guard + 1 GHz margin",
                 x=0.005, ha="left", fontsize=fs(9))
    save(fig, "figure3_wavelength_selection")


# ----------------------------------------------------------------- Figure 4
SCN_SHORT = {"nominal": "nominal", "raman_scale_0.5": "Raman x0.5",
             "raman_scale_2.0": "Raman x2",
             "raman_shape_standin_multigaussian": "stand-in shape A",
             "raman_shape_standin_oscillator": "stand-in shape B",
             "raman_level_dasilva2014": "Raman level (da Silva)",
             "attenuation_analytic_model": "analytic attenuation",
             "base_error_halved": "base error x0.5",
             "base_error_doubled": "base error x2",
             "xpm_kappa_0.25": "XPM kappa x0.25",
             "leak_isolation_minus_10dB": "isolation \u221210 dB",
             "tx_floor_x10": "Tx floor x10",
             "fwm_profile_1GHz": "FWM profile 1 GHz",
             "fwm_profile_5GHz": "FWM profile 5 GHz",
             "fwm_profile_25GHz": "FWM profile 25 GHz",
             "receiver_power_average": "receiver: power average"}
SCN_ANCHOR = "original_comparison_5km_-10dBm"


def figure4():
    m = rd("illustrative_baseline", "operating_map.csv")
    gms = rd("illustrative_baseline", "guard_margin_study.csv")
    scn = rd("illustrative_baseline", "scenario_sensitivity.csv")
    rob = rd("illustrative_baseline", "robustness_summary.csv")
    bnd = rd("illustrative_baseline", "certification_boundary.csv")
    L = np.sort(m["length_km"].unique()); P = np.sort(m["launch_dbm"].unique())

    def grid(col):
        return m.pivot(index="launch_dbm", columns="length_km",
                       values=col).reindex(index=P, columns=L).to_numpy()
    Y = grid("Y_lb_at_star").astype(float)
    LAM = grid("lambda_star_nm").astype(float)
    ok = np.array([[bool(x) for x in row] for row in grid("quality_domain_nonempty")])

    fig = plt.figure(figsize=size(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.62)
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1])
    gsc = gs[1, 0].subgridspec(2, 1, hspace=0.12)
    ax2 = fig.add_subplot(gsc[0]); ax2b = fig.add_subplot(gsc[1], sharex=ax2)
    ax3 = fig.add_subplot(gs[1, 1])
    ext = [L.min() - .5, L.max() + .5, P.min() - .5, P.max() + .5]
    grey = matplotlib.colors.ListedColormap(["#e6e6e6"])
    ax0.imshow(np.where(ok, np.nan, 1.0), extent=ext, origin="lower",
               aspect="auto", cmap=grey, vmin=0, vmax=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.log10(np.where(ok, Y, np.nan))       # undefined stays masked
    im = ax0.imshow(Z, extent=ext, origin="lower", aspect="auto", cmap="viridis",
                    vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    cb = fig.colorbar(im, ax=ax0)
    cb.set_label(r"$\log_{10}\max_{\Omega_F}Y_{\rm lb}$", fontsize=fs(7))
    cb.ax.tick_params(labelsize=fs(6))
    y_min_cert = float(np.nanmin(np.where(ok, Y, np.nan)))
    if y_min_cert < 1e-6:
        cs = ax0.contour(L, P, np.ma.masked_invalid(np.where(ok, Y, np.nan)),
                         levels=[1e-6], colors="w", linewidths=1.2)
        ax0.clabel(cs, fmt={1e-6: r"$Y_{\min}$"}, fontsize=fs(6), inline=True)
    else:
        ax0.text(0.97, 0.05, "minimum sampled optimum\n"
                 fr"$\approx {y_min_cert*1e6:.2f}\times10^{{-6}}>Y_{{\min}}$",
                 transform=ax0.transAxes, fontsize=fs(5.6), ha="right", color="0.2")
    if bnd is not None:
        bb = bnd.dropna(subset=["boundary_length_km"])
        bb = bb[bb["status"].astype(str).str.startswith("converged")]
        bb = bb.sort_values("launch_dbm")
        if len(bb):
            ax0.plot(bb["boundary_length_km"], bb["launch_dbm"], "-", color="k",
                     lw=1.3, label=r"$\max_\lambda C=F_{\min}$")
    # minimum launch power that closes the classical link, per receiver
    for Lc, Pmin, rx, ls in link_budget_lines(L):
        ax0.plot(Lc, Pmin, ls=ls, color="k", lw=1.3,
                 label=f"link budget, {rx['data_rate_gbps']:g} Gb/s")
        ax1.plot(Lc, Pmin, ls=ls, color="k", lw=1.0)
    ax0.set_ylim(P.min() - .5, P.max() + .5)
    ax1.set_ylim(P.min() - .5, P.max() + .5)
    if ax0.get_legend_handles_labels()[0]:
        ax0.legend(loc="upper right", fontsize=fs(5.6), labelcolor="0.15",
                   handlelength=2.2)
    ax0.set_xlabel("length per traversal (km)")
    ax0.set_ylabel("launch power (dBm/channel)")
    ax0.set_title(r"(a) optimized $Y_{\rm lb}$; grey: $\Omega_F$ empty",
                  loc="left", fontsize=fs(7.8))

    ax1.imshow(np.where(ok, np.nan, 1.0), extent=ext, origin="lower",
               aspect="auto", cmap=grey, vmin=0, vmax=1)
    im2 = ax1.imshow(np.where(ok, LAM, np.nan), extent=ext, origin="lower",
                     aspect="auto", cmap="plasma")
    cb2 = fig.colorbar(im2, ax=ax1)
    cb2.set_label(r"$\lambda_q^\star$ (nm)", fontsize=fs(7))
    cb2.ax.tick_params(labelsize=fs(6))
    ax1.set_xlabel("length per traversal (km)")
    ax1.set_ylabel("launch power (dBm/channel)")
    ax1.set_title(r"(b) selected wavelength; grey: undefined", loc="left",
                  fontsize=fs(7.8))

    # (c) tuning-margin sensitivity at the interior and comparison points:
    # wavelength displacement (upper) and relative yield change (lower), on two
    # separate axes rather than a dual y-axis
    # 5 km is drawn first, solid; 3 km on top, dashed with larger open markers,
    # so that coincident curves remain distinguishable
    pts = [(5.0, -10.0, "C1", "s", "-", 3.6), (3.0, -13.0, "C0", "o", "--", 5.6)]
    for L_, P_, cc, mk, lsty, msz in pts:
        g = gms[(gms["length_km"] == L_) & (gms["launch_dbm"] == P_) &
                (gms["guard_ghz"] == 50.0)].sort_values("tuning_margin_ghz")
        base = g[g["tuning_margin_ghz"] == 0.0]
        lam0 = float(base["lambda_star_nm"].iloc[0]) if len(base) else np.nan
        ax2.plot(g["tuning_margin_ghz"], g["lambda_star_nm"] - lam0, ls=lsty,
                 marker=mk, ms=ms(msz), color=cc, mfc="none",
                 label=f"{L_:g} km, {P_:+.0f} dBm/ch".replace("-", "\u2212"))
        rel = g["relative_Y_lb_change_vs_zero_margin"].where(
            g["both_optima_exist"].astype(bool))
        ax2b.plot(g["tuning_margin_ghz"], 100 * rel, ls=lsty, marker=mk,
                  ms=ms(msz), color=cc, mfc="none")
        for _, r in g.iterrows():
            if not bool(r["W_q_nonempty"]):
                ax2.plot([r["tuning_margin_ghz"]], [0], "x", color="#b71c1c",
                         ms=ms(5), mew=1.2)
    for a in (ax2, ax2b):
        a.axhline(0, color="0.6", lw=0.7, ls=":")
    ax2.set_ylabel(r"$\Delta\lambda_q^\star$ (nm)", fontsize=fs(7.5))
    ax2b.set_ylabel(r"$\Delta \max Y_{\rm lb}$ (%)", fontsize=fs(7.5))
    ax2b.set_xlabel("added tuning margin (GHz), guard 50 GHz")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_title("(c) tuning-margin sensitivity", loc="left", fontsize=fs(7.8))
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2[::-1], l2[::-1], fontsize=fs(5.8), loc="center right")
    ax2.margins(y=0.18); ax2b.margins(y=0.25)

    # (d) scenario sensitivity: M_theta with the fixed-wavelength certificate
    sc = scn[scn["anchor"] == SCN_ANCHOR].copy()
    fx = rob[(rob["anchor"] == SCN_ANCHOR) &
             (rob["mode"] == "fixed nominal wavelength")].set_index("scenario")
    sc["C_fixed"] = sc["label"].map(fx["C_bound"])
    sc = sc.sort_values("max_C_over_Omega_q")
    LO = float(min(0.985, np.nanmin(np.r_[sc["max_C_over_Omega_q"].to_numpy(float),
                                           sc["C_fixed"].to_numpy(float)]) - 0.002))
    empty = ~sc["Omega_F_nonempty"].astype(bool).to_numpy()
    cols = np.where(empty, "#b71c1c", "#2e7d32")
    yy = np.arange(len(sc))
    ax3.barh(yy, sc["max_C_over_Omega_q"] - LO, left=LO, color=cols, height=0.62)
    ax3.plot(sc["C_fixed"], yy, "D", ms=ms(3.0), mfc="white", mec="k", mew=0.8,
             ls="none", zorder=3)
    ax3.axvline(0.99, color="0.2", ls=":", lw=1.0)
    ax3.set_yticks(yy)
    ax3.set_yticklabels([SCN_SHORT.get(l, l) for l in sc["label"]], fontsize=fs(5.4))
    ax3.set_xlim(LO, min(1.0, float(np.nanmax(sc["max_C_over_Omega_q"])) + 0.002))
    ax3.tick_params(axis="x", labelsize=fs(5.8))
    ax3.set_xlabel(r"certificate $C_{7,1}(q)$")
    ax3.set_title(r"(d) scenario sensitivity, 5 km, $-10$ dBm/ch", loc="left",
                  fontsize=fs(7.8))
    ax3.text(0.9903, -0.9, r"$F_{\min}$", fontsize=fs(5.8), color="0.3")
    lg2 = ax3.legend(handles=[
        Patch(fc="#2e7d32", label=r"$M_\theta\geq F_{\min}$"),
        Patch(fc="#b71c1c", label=r"$M_\theta<F_{\min}$: no certified candidate"),
        Line2D([], [], marker="D", ms=ms(3.0), mfc="white", mec="k", ls="none",
               label="certificate at the nominal wavelength held fixed")],
        loc="upper center", fontsize=fs(5.4), ncol=1, handlelength=1.2,
        bbox_to_anchor=(0.45, -0.28))
    lg2.set_in_layout(False)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.91, bottom=0.17)
    suptitle(fig, "Figure 4 | Operating regions and sensitivity "
                 "(50 GHz guard + 1 GHz margin)", x=0.005, ha="left", fontsize=fs(9))
    save(fig, "figure4_operating_regions_and_sensitivity",
         extra=[lg2, fig._suptitle if SUPTITLE else None] + [a.title for a in (ax0, ax1, ax2, ax3)])


# ------------------------------------------------------- supplementary figure
def figure_s1():
    comp = rd("validation", "optical_composition.csv")
    pw = rd("illustrative_baseline", "fwm_profile_width.csv")
    mrs = [rd(rid, "mode_reference.csv") for rid in
           ("op_2km_m15dBm", "op_3km_m13dBm", "op_5km_m10dBm", "op_10km_m5dBm")]
    mrs = [x for x in mrs if x is not None]
    fig, ax = plt.subplots(1, 3, figsize=size(7.4, 2.6), layout="constrained")
    if comp is not None:
        err = comp["choi_trace_norm_error"].to_numpy()
        ax[0].semilogy(np.arange(len(err)), np.maximum(err, 1e-18), "o", ms=ms(3.5),
                       color="C0", mfc="none")
        ax[0].axhline(1e-9, color="0.5", ls=":", lw=0.8)
        ax[0].text(0.03, 0.88, r"target $10^{-9}$", transform=ax[0].transAxes,
                   fontsize=fs(6), color="0.35")
        ax[0].set_xlabel("three-traversal composition case")
        # "3 traversals" is spelled out: a hyphen inside math mode renders as a
        # minus sign and reads as a subtraction
        ax[0].set_ylabel(r"$\|J_{3\,\rm traversals}-J_{\rm effective}\|_1$")
        ax[0].set_ylim(1e-18, 1e-6)
    ax[0].set_title("(a) three-traversal composition", loc="left", fontsize=fs(7.6))
    if mrs:
        mr = pd.concat(mrs, ignore_index=True)
        ax[1].plot(mr["wavelength_nm"], 100 * mr["T_relative_difference"], "o",
                   ms=ms(3.5), mfc="none", color="C0", label=r"$T$")
        ax[1].plot(mr["wavelength_nm"], 100 * mr["nu_relative_difference"], "s",
                   ms=ms(3.5), mfc="none", color="C1", label=r"$\nu$")
        ax[1].plot(mr["wavelength_nm"], 100 * mr["Y_lb_relative_difference"], "^",
                   ms=ms(3.5), mfc="none", color="C3", label=r"$Y_{\rm lb}$")
        ax[1].axhline(0, color="0.6", lw=0.7, ls=":")
        ax[1].set_xlabel("wavelength (nm)")
        # the plotted quantity is (full chain - scalar) / scalar, so the label
        # names it as a relative difference rather than a bare subtraction
        ax[1].set_ylabel("relative difference from the\nscalar recurrence (%)")
        ax[1].legend(fontsize=fs(6), ncol=3, loc="center")
    ax[1].set_title("(b) receiver model: full chain vs scalar", loc="left",
                    fontsize=fs(7.6))
    if pw is not None:
        for i, (L_, P_) in enumerate([(2.0, -15.0), (3.0, -13.0), (5.0, -10.0),
                                      (10.0, -5.0)]):
            g = pw[(pw["length_km"] == L_) & (pw["launch_dbm"] == P_)].sort_values(
                "fwm_profile_rms_ghz")
            ax[2].semilogy(g["fwm_profile_rms_ghz"], g["max_nu_FWM_over_allowed"],
                           "-", marker=MARK[i % 6], ms=ms(3.4), mfc="none",
                           color=f"C{i}", label=f"{L_:g} km, {P_:+.0f} dBm".replace("-", "\u2212"))
        ax[2].set_xlabel("prescribed FWM profile rms width (GHz)")
        ax[2].set_ylabel(r"$\max_{\lambda}\sum_m\nu_{{\rm FWM},m}(\lambda)$")
        lg = ax[2].legend(fontsize=fs(5.6), ncol=2, loc="upper left",
                          bbox_to_anchor=(0.0, -0.28))
        lg.set_in_layout(False)
    ax[2].set_title("(c) FWM profile-width diagnostic", loc="left", fontsize=fs(7.6))
    suptitle(fig, "Supplementary Figure S1 | Composition check, receiver-model "
                 "comparison and FWM profile widths", x=0.005, ha="left",
                 fontsize=fs(8.5))
    save(fig, "figureS1_supplementary_diagnostics",
         extra=[a.get_legend() for a in ax] +
               [fig._suptitle if SUPTITLE else None] + [a.title for a in ax])


E0_SHORT = {0.0007: "Agnesi 2020", 0.002: "nominal", 0.004: "Bolaños 2026",
            0.007: "Xavier 2009", 0.018: "Xavier 2009 (comp.)"}
E0_LS = {0.0007: (0, (1, 1.5)), 0.002: "-", 0.004: "--", 0.007: "-.",
         0.018: (0, (5, 1.5, 1, 1.5, 1, 1.5))}


def figure_s2():
    """Base-error requirement and the classical link budget."""
    m = rd("illustrative_baseline", "operating_map.csv")
    cur = rd("illustrative_baseline", "link_budget_curve.csv")
    if m is None or "e0_max_for_certification" not in m or cur is None:
        print("  figure S2 skipped: link-budget results not found")
        return
    import config as _cf
    cfg = _cf.load("illustrative_baseline.yaml")
    anchors = sorted(float(a["e0"]) for a in cfg["base_error_sweep"]["anchors"])
    Y_min = float(cfg["targets"]["Y_min"])
    e0_nom = float(cfg["link"]["e0_per_pass"])

    L = np.sort(m["length_km"].unique()); P = np.sort(m["launch_dbm"].unique())
    E = 100 * m.pivot(index="launch_dbm", columns="length_km",
                      values="e0_max_for_certification").reindex(
                          index=P, columns=L).to_numpy().astype(float)
    fig = plt.figure(figsize=size(7.4, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.38)
    ax0 = fig.add_subplot(gs[0])
    gr = gs[1].subgridspec(2, 1, hspace=0.42)
    ax1 = fig.add_subplot(gr[0]); ax2 = fig.add_subplot(gr[1], sharex=ax1)

    # (a) largest admissible e0 over the operating map
    ext = [L.min() - .5, L.max() + .5, P.min() - .5, P.max() + .5]
    ax0.imshow(np.where(np.isfinite(E), np.nan, 1.0), extent=ext, origin="lower",
               aspect="auto", cmap=matplotlib.colors.ListedColormap(["#e6e6e6"]),
               vmin=0, vmax=1)
    im = ax0.imshow(E, extent=ext, origin="lower", aspect="auto", cmap="cividis",
                    vmin=0.0, vmax=float(np.nanmax(E)))
    cb = fig.colorbar(im, ax=ax0, pad=0.03)
    cb.set_label(r"largest admissible $e_0$ (%)", fontsize=fs(7))
    cb.ax.tick_params(labelsize=fs(6))
    lev = [100 * a for a in anchors if 100 * a < np.nanmax(E)]
    if lev:
        cs = ax0.contour(L, P, np.ma.masked_invalid(E), levels=lev, colors="w",
                         linewidths=1.0,
                         linestyles=[E0_LS.get(round(l / 100, 4), "-") for l in lev])
        cb.add_lines(cs)
    for Lc, Pmin, rx, ls in link_budget_lines(L):
        ax0.plot(Lc, Pmin, ls=ls, color="k", lw=1.3,
                 label=f"{rx['data_rate_gbps']:g} Gb/s")
    ax0.set_xlim(ext[:2]); ax0.set_ylim(ext[2:])
    ax0.set_xlabel("length per traversal (km)")
    ax0.set_ylabel("launch power (dBm/channel)")
    ax0.legend(loc="upper right", fontsize=fs(5.6), labelcolor="0.15",
               handlelength=2.2)
    ax0.set_title(r"(a) admissible $e_0$; grey: none", loc="left",
                  fontsize=fs(7.6))

    # (b) and (c) along the link-budget minimum of each receiver
    for rid, g in cur.groupby("receiver", sort=False):
        g = g.sort_values("length_km")
        col, mk = LB_COLOR.get(rid, ("0.3", "^"))
        lab = f"{float(g['data_rate_gbps'].iloc[0]):g} Gb/s"
        # the two receivers coincide at short lengths: markers are staggered
        # (odd and even km) and the second line is dashed so both stay visible
        k = list(LB_COLOR).index(rid) if rid in LB_COLOR else 0
        kw = dict(color=col, marker=mk, ms=ms(3.4), mfc="none", mew=1.0,
                  markevery=(k, 2), lw=1.2, ls=("-" if k == 0 else "--"))
        e = 100 * g["e0_max_for_certification"]
        ax1.semilogy(g["length_km"], e.where(e > 0), label=lab, **kw)
        ax2.semilogy(g["length_km"], g["Y_lb"].where(g["Y_lb"] > 0), label=lab, **kw)
    for a in anchors:
        ax1.axhline(100 * a, color="0.45", lw=0.9, ls=E0_LS.get(round(a, 4), "-"),
                    zorder=0)
    ax1.set_ylim(0.04, 3.0)
    ax1.set_xlim(L.min() - 0.5, L.max() + 0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel(r"admissible $e_0$ (%)")
    # the published anchors are named on a secondary axis, outside the plot area
    axr = ax1.twinx()
    axr.set_yscale("log"); axr.set_ylim(ax1.get_ylim())
    axr.set_yticks([100 * a for a in anchors])
    axr.set_yticklabels([E0_SHORT.get(round(a, 4), f"{100*a:g}%") for a in anchors],
                        fontsize=fs(5.4))
    axr.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axr.tick_params(axis="y", length=2.5, pad=1.5)
    ax1.set_title("(b) admissible $e_0$ at the link-budget minimum", loc="left", fontsize=fs(7.6))

    ax2.axhline(Y_min, color="#1a1a1a", ls="--", lw=1.0)
    ax2.set_xlabel("length per traversal (km)")
    ax2.set_ylabel(r"$Y_{\rm lb}(\lambda_q^\star)$")
    h, lab = ax2.get_legend_handles_labels()
    h.append(Line2D([], [], color="#1a1a1a", ls="--", lw=1.0)); lab.append(r"$Y_{\min}$")
    # one legend serves (b) and (c): the receivers keep the same colour and marker
    ax2.legend(h, lab, loc="lower left", fontsize=fs(5.6))
    ax2.set_title(fr"(c) yield bound, $e_0={100*e0_nom:g}\%$", loc="left",
                  fontsize=fs(7.6))

    fig.subplots_adjust(left=0.08, right=0.86, top=0.9, bottom=0.13)
    suptitle(fig, "Supplementary Figure S2 | Base-error requirement and the "
                 "classical link budget", x=0.005, ha="left", fontsize=fs(8.5))
    save(fig, "figureS2_base_error_and_link_budget",
         extra=[axr, fig._suptitle if SUPTITLE else None] +
               [a.title for a in (ax0, ax1, ax2)])


def build_all():
    figure1(); figure2(); figure3(); figure4(); figure_s1(); figure_s2()


if __name__ == "__main__":
    for variant in ("default", "nocaption", "final"):
        print(f"building figures from stored results ({variant}) ...")
        set_style(variant)
        build_all()
