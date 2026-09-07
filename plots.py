"""
SC2001 Project 1 - plotting utility.

    python3 plots.py            # write figures/*.png
    python3 plots.py --show     # also open interactive matplotlib windows
                                # (pan / zoom / cursor read-out)

Reads results/*.json produced by experiments.py and overlays the exact
average-case model from theory.py on top of every empirical curve.
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

from theory import hybrid_avg, pure_avg, pure_closed_form

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

C_PURE, C_H8, C_H32, C_THEORY = "#1f4e79", "#c0504d", "#2e8b57", "#999999"


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


def save(fig, name):
    path = os.path.join(FIGURES, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")


# --------------------------------------------------------------------------
# Figure 1 - fixed S, varying n
# --------------------------------------------------------------------------

def fig1(rows):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    series = {}
    for d in rows:
        key = "pure" if d["algo"] == "pure" else f"hybrid S={d['S']}"
        series.setdefault(key, []).append((d["n"], d["comparisons"]))
    for k in series:
        series[k].sort()

    colours = {"pure": C_PURE, "hybrid S=8": C_H8, "hybrid S=32": C_H32}
    for k, pts in series.items():
        ns = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        ax.plot(ns, cs, "o-", color=colours.get(k), label=k, ms=4)
        ax2.plot(ns, [c / (n * math.log2(n)) for n, c in pts],
                 "o-", color=colours.get(k), label=k, ms=4)

    ns = [p[0] for p in series["pure"]]
    ax.plot(ns, [pure_avg(n) for n in ns], "--", color=C_THEORY, lw=1.4,
            label="theory (exact recurrence)")
    ax.plot(ns, [pure_closed_form(n) for n in ns], ":", color="black", lw=1.2,
            label=r"theory  $n\log_2 n - 2n + 2$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("input size $n$")
    ax.set_ylabel("key comparisons")
    ax.set_title("Key comparisons vs $n$ (fixed $S$)")
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    ax.legend(fontsize=8)

    ax2.set_xscale("log")
    ax2.set_xlabel("input size $n$")
    ax2.set_ylabel(r"comparisons $/\ (n\log_2 n)$")
    ax2.set_title(r"Normalised: the ratio flattens $\Rightarrow\ \Theta(n\log n)$")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "fig1_comparisons_vs_n")


# --------------------------------------------------------------------------
# Figure 2 - fixed n, varying S
# --------------------------------------------------------------------------

def fig2(rows):
    n = rows[0]["n"]
    S = [d["S"] for d in rows]
    C = [d["comparisons"] for d in rows]

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax.step(S, C, where="post", color=C_PURE, lw=1.6, label="empirical")
    ax.plot(S, [hybrid_avg(n, s) for s in S], "--", color=C_THEORY, lw=1.3,
            label="theory (exact recurrence)")
    ax.axhline(C[0], color=C_H8, lw=0.9, ls=":", label="pure merge sort ($S=1$)")
    ax.set_xlabel("threshold $S$")
    ax.set_ylabel("key comparisons")
    ax.set_title(f"Key comparisons vs $S$   ($n = {n:,}$)")
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    ax.legend(fontsize=8)

    lim = 34
    axz.step(S[:lim], C[:lim], where="post", color=C_PURE, lw=1.8,
             label="empirical")
    axz.plot(S[:lim], [hybrid_avg(n, s) for s in S[:lim]], "--",
             color=C_THEORY, lw=1.3, label="theory")
    axz.set_xlabel("threshold $S$")
    axz.set_ylabel("key comparisons")
    axz.set_title("Zoom: the staircase (steps at the leaf sizes $\\approx n/2^k$)")
    axz.yaxis.set_major_formatter(FuncFormatter(millions))
    axz.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "fig2_comparisons_vs_S")


# --------------------------------------------------------------------------
# Figure 3 - optimal S across scales
# --------------------------------------------------------------------------

def fig3(rows3, rows2):
    by_n = {}
    for d in (rows3 or []):
        by_n.setdefault(d["n"], []).append((d["S"], d["comparisons"]))
    if rows2:
        by_n[rows2[0]["n"]] = [(d["S"], d["comparisons"]) for d in rows2]
    for k in by_n:
        by_n[k].sort()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    ns = sorted(by_n)
    for idx, n in enumerate(ns):
        pts = by_n[n]
        S = [p[0] for p in pts]
        C = [p[1] for p in pts]
        base = C[0]                                  # S = 1 == pure merge sort
        colour = cmap(idx / max(1, len(ns) - 1) * 0.85)
        ax.plot(S, [c / base for c in C], "-", color=colour, lw=1.5,
                label=f"n = {n:,}")
        ax2.plot(S[:24], [c / base for c in C[:24]], "o-", color=colour,
                 ms=3, lw=1.5, label=f"n = {n:,}")

    for a in (ax, ax2):
        a.axhline(1.0, color="black", lw=0.8, ls=":")
        a.set_xlabel("threshold $S$")
        a.set_ylabel("comparisons relative to pure merge sort")
        a.legend(fontsize=8)
    ax.set_title("Relative comparison cost collapses onto one curve")
    ax2.set_title("Zoom, $S \\leq 24$")

    fig.tight_layout()
    save(fig, "fig3_optimal_S")


# --------------------------------------------------------------------------
# Figure 4 - CPU time vs S
# --------------------------------------------------------------------------

def fig4(rows):
    n = rows[0]["n"]
    S = [d["S"] for d in rows]
    t_swap = [d["cpu_swap"] for d in rows]
    t_shift = [d["cpu_shift"] for d in rows]
    C = [d["comparisons"] for d in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(S, t_swap, "o-", color=C_PURE, ms=4,
            label="CPU time - swap-based leaves (lecture pseudocode)")
    ax.plot(S, t_shift, "s-", color=C_H32, ms=4,
            label="CPU time - shift-based leaves (same comparisons)")
    ax.set_xlabel("threshold $S$")
    ax.set_ylabel("CPU time (s)")
    ax.set_title(f"CPU time and comparisons pull $S$ in opposite directions "
                 f"($n = {n:,}$)")

    axc = ax.twinx()
    axc.plot(S, C, "^--", color=C_H8, ms=4, lw=1.2, label="key comparisons")
    axc.set_ylabel("key comparisons")
    axc.yaxis.set_major_formatter(FuncFormatter(millions))
    axc.grid(False)

    best_t = S[t_shift.index(min(t_shift))]
    ax.axvline(best_t, color="black", lw=0.8, ls=":")
    ax.annotate(f"time-optimal $S \\approx {best_t}$", xy=(best_t, min(t_shift)),
                xytext=(best_t + 6, min(t_shift) + 0.12), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")

    fig.tight_layout()
    save(fig, "fig4_cpu_time_vs_S")


# --------------------------------------------------------------------------
# Figure 5 - head-to-head at n = 10,000,000
# --------------------------------------------------------------------------

def fig5(rows):
    labels = [d["algo"] for d in rows]
    comps = [d["comparisons"] for d in rows]
    cpus = [d["cpu"] for d in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = [C_PURE] + [C_H32] * (len(labels) - 1)

    ax.bar(labels, comps, color=colours)
    ax.set_ylabel("key comparisons")
    ax.set_title("Key comparisons, $n = 10{,}000{,}000$")
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    for i, v in enumerate(comps):
        ax.text(i, v, f"{v/1e6:.1f}M", ha="center", va="bottom", fontsize=8)

    ax2.bar(labels, cpus, color=colours)
    ax2.set_ylabel("CPU time (s)")
    ax2.set_title("CPU time, $n = 10{,}000{,}000$")
    for i, v in enumerate(cpus):
        ax2.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)

    for a in (ax, ax2):
        a.tick_params(axis="x", rotation=20, labelsize=8)

    fig.tight_layout()
    save(fig, "fig5_head_to_head_10M")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("rendering figures")
    r1, r2, r3, r4, r5 = (load("exp1"), load("exp2"), load("exp3"),
                          load("exp4"), load("exp5"))
    if r1:
        fig1(r1)
    if r2:
        fig2(r2)
    if r2 or r3:
        fig3(r3, r2)
    if r5:
        fig4(r5)
    if r4:
        fig5(r4)
    if "--show" in sys.argv:
        plt.show()
