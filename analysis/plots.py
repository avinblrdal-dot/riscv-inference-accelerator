#!/usr/bin/env python3
"""Publication-quality figures: 300 dpi, colorblind-safe, labelled, error bars.

THESE GO ON A SCIENCE FAIR BOARD AND INTO A PAPER, so the rules are strict:

  * 300 dpi -- anything less looks blurry when printed at poster size
  * a colorblind-safe palette (Okabe-Ito). Roughly 1 in 12 men has some form
    of colour vision deficiency, and a red/green chart is unreadable to them.
    On a board being judged by a random panel, that is a real risk.
  * every axis labelled WITH UNITS. "cycles" not "value".
  * error bars wherever more than one replicate exists. A bar chart with no
    error bars claims a precision the data does not have.
  * a provenance line on every figure recording the data source and whether
    the numbers are measured, modelled or synthetic. If a figure is separated
    from its caption -- which happens constantly on a poster -- it must still
    be honest on its own.

Usage:
    python3 analysis/plots.py --csv sweep/results/synthetic_results.csv
    python3 analysis/plots.py --csv ... --out-dir analysis/figures
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from load_results import load_csv  # noqa: E402
from pareto import aggregate, energy_proxy, pareto_front  # noqa: E402

# Okabe-Ito: designed to stay distinguishable under all common forms of
# colour vision deficiency.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
             "#56B4E9", "#D55E00", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def setup(plt):
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.autolayout": True,
    })


def provenance(fig, source: str, synthetic: bool, proxy: bool,
               measured_axis: bool = True) -> None:
    """Stamp every figure with where its numbers came from.

    The distinction matters and used to be drawn too bluntly. When the model's
    weights are synthetic, ACCURACY is meaningless -- but CYCLE COUNTS are
    still real measurements, because control flow in these kernels depends
    only on tensor shapes, never on weight values. Stamping a cycles figure
    "SYNTHETIC DATA - NOT A RESULT" understated a genuine measurement just as
    badly as the opposite error would have overstated one.
    """
    bits = [f"source: {os.path.basename(source)}"]
    warn = False
    if measured_axis:
        bits.append("cycle counts MEASURED (cycle-accurate simulation)")
        if synthetic:
            bits.append("model weights synthetic - accuracy NOT meaningful")
    elif synthetic:
        bits.append("SYNTHETIC DATA - NOT A RESULT")
        warn = True
    if proxy:
        bits.append("energy axis is a PROXY, not measured")
        warn = True
    fig.text(0.005, 0.005, "  |  ".join(bits), fontsize=6.5,
             color="#B00020" if warn else "#555555",
             ha="left", va="bottom")


def fig_rq3_interaction(plt, aggs, out, source, synthetic, precision=8):
    """THE RQ3 FIGURE: cycles vs array width, one line per buffer depth.

    Plotted on a LOG y-axis, because performance effects are multiplicative:
    each factor contributes a speedup FACTOR, and on a log scale a
    multiplicative interaction appears as non-parallel lines. On a linear axis
    the same data looks parallel and the interaction is invisible.

    An earlier version plotted stall fraction, which was the right measure
    when the array was starving. With operand reuse working, the array never
    stalls (stall fraction is 0 everywhere), so cycles is the informative
    response.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sel = [a for a in aggs if int(a.get("precision", precision)) == precision] or aggs
    depths = sorted({int(a["wbuf_depth"]) for a in sel})
    widths = sorted({int(a["array_w"]) for a in sel})

    for i, d in enumerate(depths):
        xs, ys = [], []
        for w in widths:
            pts = [a for a in sel
                   if int(a["array_w"]) == w and int(a["wbuf_depth"]) == d
                   and isinstance(a.get("cycles_total"), (int, float))]
            if not pts:
                continue
            xs.append(w)
            ys.append(sum(p["cycles_total"] for p in pts) / len(pts) / 1e6)
        if xs:
            ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)],
                    color=OKABE_ITO[i % len(OKABE_ITO)],
                    linewidth=1.9, markersize=7,
                    label=f"buffer {d} words")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.get_yaxis().set_major_formatter(
        __import__("matplotlib").ticker.ScalarFormatter())
    ax.set_xlabel("MAC array width (cells per side)")
    ax.set_ylabel("Cycles per inference (millions)")
    ax.set_title("RQ3: buffer depth vs array width\n"
                 f"(int{precision}; log axes -- non-parallel lines = interaction)")
    ax.legend(title="local weight buffer", fontsize=9)
    provenance(fig, source, synthetic, False, measured_axis=True)
    path = os.path.join(out, "rq3_interaction.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_stall_interaction_unused(plt, aggs, out, source, synthetic, precision=8):
    """THE RQ3 FIGURE: stall fraction vs array width, one line per buffer depth.

    This is the single most important plot in the project. Non-parallel lines
    ARE the interaction: if buffering mattered equally at every array width the
    lines would be parallel, and the fact that they fan out is the finding.

    ONE PRECISION AT A TIME. An earlier version averaged over int8 and int4
    within each point and drew the min-max range as an error bar. That was
    wrong twice over: precision is a real experimental FACTOR, not noise, so
    collapsing it conflated a systematic effect with variability -- and the
    resulting bars extended below zero, implying negative stall fractions,
    which are impossible. Holding precision fixed keeps each line a clean
    slice through the design space.

    Error bars here show the spread ACROSS REPLICATES only. For deterministic
    RTL simulation that spread is genuinely zero, and a flat marker with no
    bar is the honest depiction -- not a defect in the plot.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sel = [a for a in aggs if int(a.get("precision", precision)) == precision]
    if not sel:
        sel = aggs
        precision = None
    depths = sorted({int(a["wbuf_depth"]) for a in sel})
    widths = sorted({int(a["array_w"]) for a in sel})

    for i, d in enumerate(depths):
        xs, ys, es = [], [], []
        for w in widths:
            pts = [a for a in sel
                   if int(a["array_w"]) == w and int(a["wbuf_depth"]) == d
                   and isinstance(a.get("stall_fraction"), (int, float))]
            if not pts:
                continue
            vals = [100 * p["stall_fraction"] for p in pts]
            xs.append(w)
            ys.append(sum(vals) / len(vals))
            # Replicate-level standard deviation, already computed by
            # aggregate(). Zero for deterministic responses, which is correct.
            es.append(100 * max(p.get("stall_fraction_sd", 0.0) for p in pts))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker=MARKERS[i % len(MARKERS)],
                        color=OKABE_ITO[i % len(OKABE_ITO)], capsize=3,
                        linewidth=1.8, markersize=6,
                        label=f"buffer depth {d}" + (" (no buffer)" if d == 0 else ""))

    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("MAC array width (cells)")
    ax.set_ylabel("Cycles stalled waiting for data (%)")
    ax.set_title("RQ3: buffering matters more as the array gets wider\n"
                 + (f"(int{precision}; non-parallel lines = interaction)"
                    if precision else "(non-parallel lines = interaction)"))
    ax.set_ylim(bottom=-2)   # stall fraction cannot be negative
    ax.legend(title="local weight buffer")
    provenance(fig, source, synthetic, False)
    path = os.path.join(out, "rq3_stall_interaction.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cycles_vs_width(plt, aggs, out, source, synthetic, precision=8):
    """Cycles vs array width, one precision at a time (see fig_stall_interaction
    for why precision is never averaged over)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sel = [a for a in aggs if int(a.get("precision", precision)) == precision] or aggs
    depths = sorted({int(a["wbuf_depth"]) for a in sel})
    widths = sorted({int(a["array_w"]) for a in sel})

    for i, d in enumerate(depths):
        xs, ys, es = [], [], []
        for w in widths:
            pts = [a for a in sel
                   if int(a["array_w"]) == w and int(a["wbuf_depth"]) == d
                   and isinstance(a.get("cycles_total"), (int, float))]
            if not pts:
                continue
            vals = [p["cycles_total"] for p in pts]
            xs.append(w)
            ys.append(sum(vals) / len(vals))
            es.append(pts[0].get("cycles_total_sd", 0.0))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker=MARKERS[i % len(MARKERS)],
                        color=OKABE_ITO[i % len(OKABE_ITO)], capsize=3,
                        linewidth=1.8, markersize=6, label=f"depth {d}")

    # Ideal scaling reference: perfect linear speedup with array width.
    base = [a for a in sel if int(a["array_w"]) == widths[0]
            and isinstance(a.get("cycles_total"), (int, float))]
    if base:
        c0 = sum(b["cycles_total"] for b in base) / len(base)
        ax.plot(widths, [c0 * widths[0] / w for w in widths], "k--",
                linewidth=1.0, alpha=0.6, label="ideal linear scaling")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("MAC array width (cells)")
    ax.set_ylabel("Cycles per inference")
    ax.set_title(f"Cycles vs array width (int{precision})\n"
                 "gap from the dashed line = the cost of starving the array")
    ax.legend(fontsize=9)
    provenance(fig, source, synthetic, False)
    path = os.path.join(out, "cycles_vs_width.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pareto(plt, aggs, out, source, synthetic, using_proxy):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    front = pareto_front(aggs, [("energy", "min"), ("accuracy", "max"),
                                ("luts", "min")])
    front_ids = {id(p) for p in front}

    precisions = sorted({int(a["precision"]) for a in aggs})
    for i, prec in enumerate(precisions):
        pts = [a for a in aggs if int(a["precision"]) == prec
               and a.get("energy") is not None
               and isinstance(a.get("accuracy"), (int, float))]
        dom = [p for p in pts if id(p) not in front_ids]
        if dom:
            ax.scatter([p["energy"] for p in dom], [100*p["accuracy"] for p in dom],
                       s=26, alpha=0.30, color=OKABE_ITO[i % len(OKABE_ITO)],
                       marker=MARKERS[i % len(MARKERS)],
                       label=f"int{prec} (dominated)")
        nd = [p for p in pts if id(p) in front_ids]
        if nd:
            ax.scatter([p["energy"] for p in nd], [100*p["accuracy"] for p in nd],
                       s=95, color=OKABE_ITO[i % len(OKABE_ITO)],
                       marker=MARKERS[i % len(MARKERS)], edgecolors="black",
                       linewidths=0.9, label=f"int{prec} (Pareto optimal)")

    fr = sorted([p for p in front if p.get("energy") is not None],
                key=lambda p: p["energy"])
    if fr:
        ax.plot([p["energy"] for p in fr], [100*p["accuracy"] for p in fr],
                "k-", linewidth=1.0, alpha=0.5, zorder=0)

    xlabel = ("Energy proxy (cycles x LUTs / 1e6) - NOT MEASURED"
              if using_proxy else "Energy per inference (uJ, measured)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("Energy / accuracy trade-off\nlarge outlined markers are Pareto optimal")
    ax.legend(fontsize=8.5, loc="lower right")
    provenance(fig, source, synthetic, using_proxy, measured_axis=False)
    path = os.path.join(out, "pareto_energy_accuracy.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_area(plt, aggs, out, source, synthetic):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    widths = sorted({int(a["array_w"]) for a in aggs})
    precisions = sorted({int(a["precision"]) for a in aggs})
    w_bar = 0.36

    for i, prec in enumerate(precisions):
        xs, luts, lerr, dsps = [], [], [], []
        for j, w in enumerate(widths):
            pts = [a for a in aggs if int(a["array_w"]) == w
                   and int(a["precision"]) == prec
                   and isinstance(a.get("luts"), (int, float))]
            if not pts:
                continue
            xs.append(j + (i - (len(precisions)-1)/2) * w_bar)
            luts.append(sum(p["luts"] for p in pts) / len(pts))
            lerr.append(max(p.get("luts_sd", 0.0) for p in pts))
            dsps.append(sum(p.get("dsps", 0) for p in pts) / len(pts))
        if xs:
            ax1.bar(xs, luts, w_bar, yerr=lerr, capsize=3,
                    color=OKABE_ITO[i % len(OKABE_ITO)], label=f"int{prec}")
            ax2.bar(xs, dsps, w_bar, color=OKABE_ITO[i % len(OKABE_ITO)],
                    label=f"int{prec}")

    for ax, ylab, title in ((ax1, "LUTs", "Logic area"),
                            (ax2, "DSP slices", "Hardened multipliers")):
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlabel("MAC array width (cells)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=9)

    # The Arty A7-100T has 240 DSP slices -- the reason the 100T was chosen.
    ax2.axhline(240, color="#D55E00", linestyle="--", linewidth=1.1)
    ax2.text(0.02, 245, "Arty A7-100T budget: 240 DSPs", fontsize=8,
             color="#D55E00")

    provenance(fig, source, synthetic, False)
    path = os.path.join(out, "area_vs_width.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(ROOT, "sweep", "results",
                                                  "synthetic_results.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--workload", default="workload_a")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")   # no display needed; works headless and in CI
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is not installed.\n"
              "  pip install matplotlib", file=sys.stderr)
        return 2

    setup(plt)
    rows = load_csv(args.csv)
    rows = [r for r in rows if r.get("workload") == args.workload]
    if not rows:
        print(f"No usable rows for workload={args.workload}", file=sys.stderr)
        return 1

    aggs = aggregate(rows, ("array_w", "array_h", "wbuf_depth", "precision",
                            "workload"))
    using_proxy = not any(
        isinstance(a.get("energy_per_inference_uj"), (int, float)) for a in aggs)
    for a in aggs:
        a["energy"] = (energy_proxy(a) if using_proxy
                       else a.get("energy_per_inference_uj"))

    synthetic = any(a.get("accuracy_is_synthetic") for a in aggs)
    os.makedirs(args.out_dir, exist_ok=True)

    made = [
        fig_rq3_interaction(plt, aggs, args.out_dir, args.csv, synthetic),
        fig_cycles_vs_width(plt, aggs, args.out_dir, args.csv, synthetic),
    ]
    # Accuracy and area come from a trained model and from Vivado. Skip those
    # figures rather than drawing empty axes when the data does not exist.
    if any(isinstance(a.get("accuracy"), (int, float)) for a in aggs) and \
       any(isinstance(a.get("luts"), (int, float)) for a in aggs):
        made.append(fig_pareto(plt, aggs, args.out_dir, args.csv, synthetic,
                               using_proxy))
        made.append(fig_area(plt, aggs, args.out_dir, args.csv, synthetic))
    else:
        print("  (skipping Pareto and area figures -- no accuracy/synthesis"
              " data yet)")

    print("Wrote:")
    for p in made:
        print(f"  {p}")
    if synthetic or using_proxy:
        print()
        print("  Every figure is stamped with its provenance. These are built")
        print("  from synthetic and/or proxy data and are NOT results -- they")
        print("  exist to prove the plotting pipeline works before real data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
