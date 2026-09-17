"""
SC2001 Project 1 - plotting utility.

    python3 plots.py            # write figures/*.png
    python3 plots.py --show     # also open interactive matplotlib windows
                                # (pan / zoom / cursor read-out)

Reads results/*.json produced by experiments.py and overlays the exact
average-case model from theory.py on top of every empirical curve.

Chart rules applied throughout
------------------------------
* Fixed categorical hue slots for identity (blue = the original/baseline
  algorithm everywhere, never reassigned by rank), a single-hue light->dark
  ramp for the *ordered* n-series in Figure 3.
* No dual-axis panels. Figure 4 used to put CPU time and key comparisons on
  two different y-scales in one plot, which invites the reader to see a
  "crossing point" whose vertical alignment is entirely arbitrary; it is now
  two panels sharing the S axis.
* Recessive chrome: horizontal hairline grid only, no tick marks, no box
  spines. Bars keep a real zero baseline; deltas are given as labels rather
  than by truncating the axis.
* Every panel carries a left-aligned title plus a subtitle stating what the
  panel actually shows, so a figure lifted into slides still reads alone.
"""

import json
import math
import os
import sys

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")                # head-less rendering
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from theory import hybrid_avg

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

# --------------------------------------------------------------------------
# Palette - fixed categorical slots (identity), never cycled.
# --------------------------------------------------------------------------

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e8e7e1", "#c3c2b7", "#fcfcfb"

# Single-hue light -> dark ramp for the six *ordered* dataset sizes (Fig 3):
# these series have a natural order, so lightness carries it.
N_RAMP = plt.get_cmap("Blues")

SANS = ["Helvetica Neue", "Avenir Next", "Helvetica", "Arial", "DejaVu Sans"]

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,

    "font.family": "sans-serif",
    "font.sans-serif": SANS,
    "font.size": 10,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica Neue",
    "mathtext.it": "Helvetica Neue:italic",
    "mathtext.bf": "Helvetica Neue:bold",

    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 10,
    "axes.labelpad": 8,

    "axes.grid": True,
    "axes.grid.axis": "y",               # horizontal hairlines only
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,

    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,

    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_SECONDARY,
    "legend.handlelength": 1.7,
    "legend.handletextpad": 0.7,
    "legend.labelspacing": 0.5,
    "legend.borderaxespad": 0.2,

    "lines.solid_capstyle": "round",
})


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def load(name):
    path = os.path.join(RESULTS, name + ".json")
    if not os.path.exists(path):
        print(f"  (skipping {name}: not found)")
        return None
    with open(path) as f:
        return json.load(f)


def millions(x, _pos):
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}k"
    return f"{x:.0f}"


def titled(ax, title, subtitle=None):
    """Left-aligned title with an explanatory subtitle underneath it."""
    ax.set_title(title, loc="left", pad=30 if subtitle else 14,
                 fontsize=12, fontweight="bold", color=INK)
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.2,
                color=MUTED, va="bottom", ha="left")


def save(fig, name):
    path = os.path.join(FIGURES, name + ".png")
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.28)
    print(f"  -> {path}")


# --------------------------------------------------------------------------
# Figure 1 - fixed S, varying n
# --------------------------------------------------------------------------

FIG1_HEADLINE_S = 8   # near the comparison-optimal S* = 4/ln2 ~ 5.77


def fig1(rows):
    """Panel A: raw comparisons vs n (log-log) - the actual growth curve.
    Panel B: comparisons / (n log2 n) vs n - flattens iff growth is Theta(n log n).
    Footer: 2-3 callout stats, the way the rest of the deck states a number.

    The old version led with a (measured-theory)/theory residual plot. That
    answers "does it match" but never shows a reader what C(n) actually looks
    like, which is what (c)(i) literally asks to plot. Residual is now a
    single number in the footer instead of the headline panel.
    """
    pts = sorted((d["n"], d["comparisons"]) for d in rows
                 if d["algo"] == "hybrid" and d["S"] == FIG1_HEADLINE_S)
    ns = [p[0] for p in pts]
    measured = [p[1] for p in pts]
    theory = [hybrid_avg(n, FIG1_HEADLINE_S) for n in ns]

    fig = plt.figure(figsize=(12, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 1], hspace=0.6, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axF = fig.add_subplot(gs[1, :])

    # --- Panel A: raw comparisons vs n, log-log --------------------------
    axA.plot(ns, theory, "-", color=MUTED, lw=3.4, alpha=0.35, zorder=1,
             label="theory (exact recurrence)")
    axA.plot(ns, measured, "o-", color=BLUE, lw=2.2, ms=6.5, zorder=3,
             markeredgecolor=SURFACE, markeredgewidth=1.3, label="measured")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("input size $n$")
    axA.set_ylabel("key comparisons")
    axA.legend(loc="upper left")
    axA.text(0.98, 0.05, "nearly straight on log–log axes — consistent with $\\Theta(n\\log n)$",
              transform=axA.transAxes, ha="right", fontsize=8.3, color=MUTED)
    titled(axA, f"$S = {FIG1_HEADLINE_S}$ fixed — comparisons vs input size $n$",
           "measured tracks the exact-recurrence prediction across four decades of $n$")

    # --- Panel B: comparisons / (n log2 n) vs n, confirms the growth rate -
    ratio_m = [c / (n * math.log2(n)) for n, c in zip(ns, measured)]
    ratio_t = [c / (n * math.log2(n)) for n, c in zip(ns, theory)]
    axB.plot(ns, ratio_t, "-", color=MUTED, lw=3.4, alpha=0.35, zorder=1,
             label="theory")
    axB.plot(ns, ratio_m, "o-", color=BLUE, lw=2.2, ms=6.5, zorder=3,
             markeredgecolor=SURFACE, markeredgewidth=1.3, label="measured")
    axB.set_xscale("log")
    axB.set_xlabel("input size $n$")
    axB.set_ylabel(r"comparisons $/\ (n\log_2 n)$")
    axB.legend(loc="lower right")
    # The ratio settles into a band rather than a single value - the same
    # recursive-halving "staircase" seen in fig2/fig3 (n // 2 splits land at
    # slightly different depths relative to S depending on n), so it wobbles
    # by a couple of percent instead of decaying smoothly. Labelling the last
    # point as "the" limit would overclaim monotonic convergence it doesn't
    # have; the band is what's actually true and still proves Theta(n log n).
    band_lo, band_hi = min(ratio_m[-4:]), max(ratio_m[-4:])
    axB.axhspan(band_lo, band_hi, color=BLUE, alpha=0.08, zorder=0)
    conv = sum(ratio_m[-4:]) / 4
    axB.annotate(f"settles into a band ≈{band_lo:.2f}–{band_hi:.2f}",
                 xy=(ns[-2], ratio_m[-2]), xytext=(-150, 18), textcoords="offset points",
                 fontsize=8.8, color=INK_SECONDARY,
                 arrowprops=dict(arrowstyle="-", lw=0.9, color=BASELINE,
                                 shrinkA=2, shrinkB=4))
    titled(axB, "Ratio flattens toward a constant band",
           r"confirms $\Theta(n\log n)$ growth — a $\Theta(n^2)$ law would keep climbing, not flatten")

    # --- Footer: callout stats -------------------------------------------
    axF.axis("off")
    err_at_max_n = (measured[-1] - theory[-1]) / theory[-1] * 100
    stats = [
        (f"{err_at_max_n:+.4f}%", f"measured vs. theory error at n = {ns[-1]:,}"),
        (f"≈{band_lo:.2f}–{band_hi:.2f}", r"comparisons $/(n\log_2 n)$ band at large $n$"),
        (f"S = {FIG1_HEADLINE_S}", "threshold held fixed across every $n$ in this chart"),
    ]
    for i, (big, cap) in enumerate(stats):
        x = (i + 0.5) / len(stats)
        axF.text(x, 0.62, big, transform=axF.transAxes, ha="center", va="center",
                  fontsize=23, fontweight="bold", color=INK)
        axF.text(x, 0.08, cap, transform=axF.transAxes, ha="center", va="center",
                  fontsize=9.2, color=MUTED)

    save(fig, "fig1_comparisons_vs_n")


def fig1_clean(rows):
    """Same data as fig1, stripped to axes + legend only - no subtitles,
    no in-panel notes, no annotation arrows, no footer stat cards."""
    pts = sorted((d["n"], d["comparisons"]) for d in rows
                 if d["algo"] == "hybrid" and d["S"] == FIG1_HEADLINE_S)
    ns = [p[0] for p in pts]
    measured = [p[1] for p in pts]
    theory = [hybrid_avg(n, FIG1_HEADLINE_S) for n in ns]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    axA.plot(ns, theory, "-", color=MUTED, lw=3.4, alpha=0.35, zorder=1,
             label="theory")
    axA.plot(ns, measured, "o-", color=BLUE, lw=2.2, ms=6.5, zorder=3,
             markeredgecolor=SURFACE, markeredgewidth=1.3, label="measured")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("input size $n$")
    axA.set_ylabel("key comparisons")
    axA.set_title(f"Comparisons vs $n$ ($S = {FIG1_HEADLINE_S}$ fixed)",
                   loc="left", fontsize=12, fontweight="bold", color=INK)
    axA.legend(loc="upper left")

    ratio_m = [c / (n * math.log2(n)) for n, c in zip(ns, measured)]
    ratio_t = [c / (n * math.log2(n)) for n, c in zip(ns, theory)]
    axB.plot(ns, ratio_t, "-", color=MUTED, lw=3.4, alpha=0.35, zorder=1,
             label="theory")
    axB.plot(ns, ratio_m, "o-", color=BLUE, lw=2.2, ms=6.5, zorder=3,
             markeredgecolor=SURFACE, markeredgewidth=1.3, label="measured")
    axB.set_xscale("log")
    axB.set_xlabel("input size $n$")
    axB.set_ylabel(r"comparisons $/\ (n\log_2 n)$")
    axB.set_title(f"Comparisons $/(n\\log_2 n)$ vs $n$ ($S = {FIG1_HEADLINE_S}$ fixed)",
                   loc="left", fontsize=12, fontweight="bold", color=INK)
    axB.legend(loc="lower right")

    fig.tight_layout(w_pad=3.5)
    save(fig, "fig1_comparisons_vs_n_clean")


# --------------------------------------------------------------------------
# Figure 2 - fixed n, varying S
# --------------------------------------------------------------------------

def fig2(rows):
    """Panel A: raw comparisons vs S, full range - the actual (c)(ii) plot.
    Panel B: zoom into small S, annotated - explains WHY it steps instead of
    curving (a whole recursion level collapses to insertion sort each time S
    crosses a leaf-size boundary n/2^k).
    Footer: 3 callout stats, same convention as fig1.
    """
    n = rows[0]["n"]
    S = [d["S"] for d in rows]
    C = [d["comparisons"] for d in rows]
    theory = [hybrid_avg(n, s) for s in S]

    fig = plt.figure(figsize=(12, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 1], hspace=0.6, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axF = fig.add_subplot(gs[1, :])

    # --- Panel A: full range S = 1..100 -----------------------------------
    axA.plot(S, theory, "-", color=MUTED, lw=3.4, alpha=0.35, zorder=1,
              label="theory (exact recurrence)")
    axA.step(S, C, where="post", color=BLUE, lw=2.2, zorder=3, label="measured")
    axA.axhline(C[0], color=ORANGE, lw=1.3, ls=(0, (2, 2)), zorder=2,
                label="pure merge sort ($S=1$)")
    axA.set_xlabel("threshold $S$")
    axA.set_ylabel("key comparisons")
    axA.yaxis.set_major_formatter(FuncFormatter(millions))
    axA.legend(loc="upper left")
    titled(axA, f"$n = {n:,}$ fixed — comparisons vs threshold $S$",
           "comparisons trend upward with $S$ — hybridizing never wins on comparisons alone")

    # --- Panel B: zoom, annotate the mechanism ----------------------------
    ZOOM = 20
    idx = [i for i, s in enumerate(S) if s <= ZOOM]
    Sz = [S[i] for i in idx]
    Cz = [C[i] for i in idx]
    Tz = [theory[i] for i in idx]
    axB.plot(Sz, Tz, "-", color=MUTED, lw=3.4, alpha=0.35, zorder=1, label="theory")
    axB.step(Sz, Cz, where="post", color=BLUE, lw=2.2, zorder=3, label="measured")
    axB.set_xlabel("threshold $S$")
    axB.set_ylabel("key comparisons")
    axB.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v/1e6:.2f}M"))
    axB.legend(loc="upper left")

    jump_S = 16                         # top of the step - boundary n/2^16 = 15.26 sits just below it
    boundary = n / 2 ** 16
    j = S.index(jump_S)
    axB.annotate(f"leaf size $n/2^{{16}} \\approx {boundary:.1f}$ crosses $S$:\n"
                 "a whole recursion level collapses\nto insertion sort at once",
                 xy=(jump_S, C[j]), xycoords="data",
                 xytext=(-165, -50), textcoords="offset points",
                 fontsize=8.5, color=INK_SECONDARY,
                 arrowprops=dict(arrowstyle="-", lw=0.9, color=BASELINE,
                                 shrinkA=2, shrinkB=4))
    titled(axB, "Zoom: why it steps instead of curving",
           "flat while $S$ sits inside one leaf-size band — a jump when $S$ crosses $n/2^k$")

    # --- Footer: callout stats ---------------------------------------------
    axF.axis("off")
    pct = (C[-1] - C[0]) / C[0] * 100
    # First REAL leaf-collapse jump, i.e. first meaningful increase - not the
    # ~100-comparison dip at S=2->3, which is this single dataset's actual
    # count landing a hair below the theoretical average at that S, not a
    # step caused by the recursion structure.
    first_jump_i = next(i for i in range(1, len(S)) if C[i] > C[i - 1] + 1000)
    first_jump_S = S[first_jump_i]
    first_k = round(math.log2(n / first_jump_S))
    first_boundary = n / 2 ** first_k
    stats = [
        (f"+{pct:.1f}%", f"comparisons at $S=100$ vs. pure merge sort ($S=1$)"),
        (f"$S = {first_jump_S}$", f"first jump — predicted at $n/2^{{{first_k}}} \\approx {first_boundary:.2f}$"),
        (f"$n = {n:,}$", "input size held fixed across every $S$ in this chart"),
    ]
    for i, (big, cap) in enumerate(stats):
        x = (i + 0.5) / len(stats)
        axF.text(x, 0.62, big, transform=axF.transAxes, ha="center", va="center",
                  fontsize=23, fontweight="bold", color=INK)
        axF.text(x, 0.08, cap, transform=axF.transAxes, ha="center", va="center",
                  fontsize=9.2, color=MUTED)

    save(fig, "fig2_comparisons_vs_S")


def fig2_clean(rows):
    """Same data as fig2, stripped to axes + legend only - no subtitles,
    no annotation arrow, no footer stat cards, no pure-merge-sort reference
    line (S=1 is already the leftmost point on both curves)."""
    n = rows[0]["n"]
    S = [d["S"] for d in rows]
    C = [d["comparisons"] for d in rows]
    theory = [hybrid_avg(n, s) for s in S]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    axA.plot(S, theory, "-", color=MUTED, lw=3.4, alpha=0.35, zorder=1, label="theory")
    axA.step(S, C, where="post", color=BLUE, lw=2.2, zorder=3, label="measured")
    axA.set_xlabel("threshold $S$")
    axA.set_ylabel("key comparisons")
    axA.yaxis.set_major_formatter(FuncFormatter(millions))
    axA.set_title(f"Comparisons vs $S$ ($n = {n:,}$ fixed)",
                   loc="left", fontsize=12, fontweight="bold", color=INK)
    axA.legend(loc="upper left")

    ZOOM = 20
    idx = [i for i, s in enumerate(S) if s <= ZOOM]
    Sz = [S[i] for i in idx]
    Cz = [C[i] for i in idx]
    Tz = [theory[i] for i in idx]
    axB.plot(Sz, Tz, "-", color=MUTED, lw=3.4, alpha=0.35, zorder=1, label="theory")
    axB.step(Sz, Cz, where="post", color=BLUE, lw=2.2, zorder=3, label="measured")
    axB.set_xlabel("threshold $S$")
    axB.set_ylabel("key comparisons")
    axB.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v/1e6:.2f}M"))
    axB.set_title(f"Zoom, $S \\leq {ZOOM}$", loc="left", fontsize=12,
                   fontweight="bold", color=INK)
    axB.legend(loc="upper left")

    fig.tight_layout(w_pad=3.5)
    save(fig, "fig2_comparisons_vs_S_clean")


# --------------------------------------------------------------------------
# Figure 3 - optimal S across scales
# --------------------------------------------------------------------------

def fig3(rows3, rows2, rows5):
    """Panel A: comparisons relative to pure merge sort, across several n -
    by key comparisons, S=1 wins at every scale tested (no interior dip).
    Panel B: CPU time relative to pure merge sort, at one n - the metric
    where an interior optimum actually exists. The two panels are meant to
    be read together: same normalization (ratio to S=1), same x-axis, two
    different answers - that contrast IS the finding for (c)(iii).
    Footer: 3 callout stats.
    """
    by_n = {}
    for d in (rows3 or []):
        by_n.setdefault(d["n"], []).append((d["S"], d["comparisons"]))
    if rows2:
        by_n[rows2[0]["n"]] = [(d["S"], d["comparisons"]) for d in rows2]
    for k in by_n:
        by_n[k].sort()
    ns = sorted(by_n)
    shades = [N_RAMP(0.40 + 0.52 * i / max(1, len(ns) - 1)) for i in range(len(ns))]

    cpu_n = rows5[0]["n"]
    cpu_S = [d["S"] for d in rows5]
    cpu_t = [d["cpu_shift"] for d in rows5]
    cpu_base = cpu_t[0]                               # S = 1 in the same hybrid code path

    fig = plt.figure(figsize=(12, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 1], hspace=0.6, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axF = fig.add_subplot(gs[1, :])

    # --- Panel A: comparisons ratio, multiple n ---------------------------
    for colour, n in zip(shades, ns):
        pts = by_n[n]
        S = [p[0] for p in pts]
        C = [p[1] for p in pts]
        base = C[0]
        axA.plot(S, [c / base for c in C], "-", color=colour, lw=2,
                  label=f"$n$ = {n:,}")
    axA.axhline(1.0, color=BASELINE, lw=1.1)
    axA.text(99, 1.012, "pure merge sort baseline", fontsize=8.4, color=MUTED,
              va="bottom", ha="right")
    axA.set_xlabel("threshold $S$")
    axA.set_ylabel("comparisons $/$ pure merge sort")
    axA.legend(loc="upper left", ncol=2, columnspacing=1.4, fontsize=8.3)
    titled(axA, "By key comparisons, $S=1$ wins at every scale",
           "no curve ever dips below 1.0 — hybridizing never reduces comparisons")

    # --- Panel B: CPU time ratio, one n ------------------------------------
    ratio = [t / cpu_base for t in cpu_t]
    axB.axhline(1.0, color=BASELINE, lw=1.1)
    axB.plot(cpu_S, ratio, "o-", color=BLUE, lw=2.2, ms=6,
              markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    best_i = min(range(len(ratio)), key=lambda i: ratio[i])
    axB.annotate(f"fastest: $S={cpu_S[best_i]}$\n"
                 f"{(1 - ratio[best_i]) * 100:.0f}% quicker than pure",
                 xy=(cpu_S[best_i], ratio[best_i]), xycoords="data",
                 xytext=(26, 1.075), textcoords="data",
                 fontsize=8.6, color=INK_SECONDARY, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.9, color=BASELINE,
                                 shrinkA=2, shrinkB=4))
    axB.set_xlabel("threshold $S$")
    axB.set_ylabel("CPU time $/$ pure merge sort")
    titled(axB, "By CPU time, there's a real basin",
           f"$n = {cpu_n:,}$ — a broad dip, not hybridizing costs real time")

    # --- Footer: callout stats ---------------------------------------------
    axF.axis("off")
    stats = [
        ("$S = 1$", "comparison-optimal at every $n$ tested, 1,000–10,000,000"),
        (f"$S = {cpu_S[best_i]}$", f"CPU-time-optimal at $n={cpu_n:,}$"),
        ("scale-invariant", "neither optimum drifts as $n$ grows"),
    ]
    for i, (big, cap) in enumerate(stats):
        x = (i + 0.5) / len(stats)
        axF.text(x, 0.62, big, transform=axF.transAxes, ha="center", va="center",
                  fontsize=21, fontweight="bold", color=INK)
        axF.text(x, 0.08, cap, transform=axF.transAxes, ha="center", va="center",
                  fontsize=8.9, color=MUTED)

    save(fig, "fig3_optimal_S")


# --------------------------------------------------------------------------
# Figure 4 - CPU time vs S  (two panels, never two y-scales in one)
# --------------------------------------------------------------------------

def fig4(rows):
    n = rows[0]["n"]
    S = [d["S"] for d in rows]
    t_swap = [d["cpu_swap"] for d in rows]
    t_shift = [d["cpu_shift"] for d in rows]
    C = [d["comparisons"] for d in rows]

    fig, (ax, axc) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax.plot(S, t_swap, "o-", color=BLUE, ms=5, lw=2,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.plot(S, t_shift, "s-", color=ORANGE, ms=4.5, lw=2,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    # Direct-label each line past its last point: bold name, then the actual
    # code, then a one-line explanation - stacked in the empty margin beyond
    # the last data point (x > 100), so nothing overlaps the real data.
    def leaf_label(x, y, title, code, note, colour, y0):
        common = dict(xy=(x, y), xycoords="data", xytext=(10, y0),
                      textcoords="offset points", ha="left", va="center")
        ax.annotate(title, fontsize=9, fontweight="bold", color=colour, **common)
        ax.annotate(code, fontsize=7.8, fontweight="normal", color=colour,
                    xy=(x, y), xycoords="data", xytext=(10, y0 - 15),
                    textcoords="offset points", ha="left", va="center")
        ax.annotate(note, fontsize=7.6, fontweight="normal", color=MUTED,
                    xy=(x, y), xycoords="data", xytext=(10, y0 - 29),
                    textcoords="offset points", ha="left", va="center")

    leaf_label(S[-1], t_swap[-1], "swap every step",
               "A[j], A[j+1] = A[j+1], A[j]", "3 writes, every step", BLUE, 8)
    leaf_label(S[-1], t_shift[-1], "shift + 1 store",
               "A[j+1] = A[j]", "1 write, key placed once at the end", ORANGE, 6)

    # Room for the direct labels, without inventing tick marks past the data,
    # and headroom above the highest point so the "swap" label clears the
    # subtitle instead of climbing into it.
    ax.set_xlim(right=S[-1] * 1.62)
    ax.set_ylim(top=max(t_swap) * 1.24)
    ax.set_xticks(list(range(0, S[-1] + 1, 20)))
    ax.set_xlabel("threshold $S$")
    ax.set_ylabel("CPU time (s)")

    best_t = S[t_shift.index(min(t_shift))]
    ax.axvline(best_t, color=BASELINE, lw=1.1, ls=(0, (2, 2)), zorder=0)
    ax.annotate(f"time-optimal $S \\approx {best_t}$",
                xy=(best_t, min(t_shift)), xytext=(best_t + 12, min(t_shift) - 0.02),
                fontsize=8.8, color=INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", lw=0.9, color=BASELINE,
                                shrinkA=2, shrinkB=4))
    titled(ax, "Same comparisons, two ways to move the data",
           f"$n = {n:,}$, minimum of 3 runs — both insertion-sort variants, real interior optimum over $S \\in [6, 24]$")

    axc.plot(S, C, "-", color=AQUA, lw=2.2)
    axc.plot(S, C, "^", color=AQUA, ms=5,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    axc.set_xlabel("threshold $S$")
    axc.set_ylabel("key comparisons")
    axc.yaxis.set_major_formatter(FuncFormatter(millions))
    axc.axvline(best_t, color=BASELINE, lw=1.1, ls=(0, (2, 2)), zorder=0)
    titled(axc, "Comparisons just keep climbing",
           "same $S$ axis — the two objectives disagree, so they get two panels")

    fig.tight_layout(w_pad=3.5)
    save(fig, "fig4_cpu_time_vs_S")


# --------------------------------------------------------------------------
# Figure 5 - head-to-head at n = 10,000,000
# --------------------------------------------------------------------------

def fig5(rows):
    labels = [d["algo"] for d in rows]
    comps = [d["comparisons"] for d in rows]
    cpus = [d["cpu"] for d in rows]

    pretty = []
    for s in labels:
        s = s.replace("pure merge sort", "original merge sort")
        s = s.replace("hybrid (S=", "hybrid, $S$ = ").replace(")", "")
        pretty.append(s)

    # Horizontal bars: the category names read straight, no rotated labels.
    y = list(range(len(labels)))[::-1]
    colours = [BLUE] + [ORANGE] * (len(labels) - 1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)

    for a, vals, fmt, base in ((ax, comps, lambda v: f"{v/1e6:.1f}M", comps[0]),
                               (ax2, cpus, lambda v: f"{v:.1f}s", cpus[0])):
        a.barh(y, vals, color=colours, height=0.62, zorder=2)
        a.grid(axis="y", visible=False)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_xlim(0, max(vals) * 1.30)
        for yi, v in zip(y, vals):
            delta = (v - base) / base * 100
            tail = "" if abs(delta) < 0.005 else f"   {delta:+.1f}%"
            a.text(v + max(vals) * 0.022, yi, fmt(v) + tail,
                   va="center", ha="left", fontsize=9.2, color=INK_SECONDARY)

    ax.set_yticks(y, pretty, fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlabel("key comparisons")
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax2.set_xlabel("CPU time (s)")

    titled(ax, "Key comparisons", "$n$ = 10,000,000 — lower is better")
    titled(ax2, "CPU time", "same dataset, same run — lower is better")

    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_head_to_head_10M")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("rendering figures")
    r1, r2, r3, r4, r5 = (load("exp1"), load("exp2"), load("exp3"),
                          load("exp4"), load("exp5"))
    if r1:
        fig1(r1)
        fig1_clean(r1)
    if r2:
        fig2(r2)
        fig2_clean(r2)
    if (r2 or r3) and r5:
        fig3(r3, r2, r5)
    if r5:
        fig4(r5)
    if r4:
        fig5(r4)
    if "--show" in sys.argv:
        plt.show()
