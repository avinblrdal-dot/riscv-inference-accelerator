#!/usr/bin/env python3
"""Figures for the matrix-multiply benchmark (docs/DECISIONS.md D024).

Reads sweep/results/matmul_results.csv (written by sweep/run_matmul_sweep.py)
and writes:

  analysis/figures/matmul_speedup_vs_k.png
      Speedup vs. dot-product length K on the 4x4 array, one line per buffer
      depth, plus DOT4. The drop where K first exceeds the buffer is the
      answer to "at what chunk size does the speedup stop improving".

  analysis/figures/matmul_width.png
      Best-case speedup (K = 1024, 1024-word buffer) by array width.

Only rows with result_correct == True are plotted. Exploratory experiment:
descriptive figures only, no significance tests.

Usage:  python3 analysis/plot_matmul.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_IN = os.path.join(ROOT, "sweep", "results", "matmul_results.csv")


def load() -> list[dict]:
    with open(CSV_IN) as fh:
        rows = [r for r in csv.DictReader(fh) if r["result_correct"] == "True"]
    for r in rows:
        for k in ("m", "n", "k", "cycles"):
            r[k] = int(r[k])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load()

    base = {r["k"]: r["cycles"] for r in rows
            if r["exp"] == "ksweep" and r["impl"] == "baseline"}
    dot4 = {r["k"]: r["cycles"] for r in rows
            if r["exp"] == "ksweep" and r["impl"] == "dot4"}
    ks = sorted(base)

    # --- speedup vs K, 4x4 array -------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    colors = {16: "#c0392b", 64: "#e67e22", 256: "#2980b9", 1024: "#27ae60"}
    for depth in (16, 64, 256, 1024):
        pts = sorted((r["k"], base[r["k"]] / r["cycles"]) for r in rows
                     if r["exp"] == "ksweep" and r["impl"] == "array"
                     and r["array_w"] == "4" and r["buf_depth"] == str(depth))
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                color=colors[depth], label=f"4x4 array, {depth}-word buffer")
        if depth < max(ks):
            ax.axvline(depth, color=colors[depth], ls=":", lw=1, alpha=0.6)
    ax.plot(ks, [base[k] / dot4[k] for k in ks], marker="s", color="#7f8c8d",
            ls="--", label="DOT4 instruction (no buffer)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Dot-product length K (values per output)")
    ax.set_ylabel("Speedup vs. plain RISC-V (x)")
    ax.set_title("Speedup climbs until the data outgrows the buffer, then drops\n"
                 "(16x16 output; dotted lines mark each buffer's size)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "matmul_speedup_vs_k.png")
    fig.savefig(p1)
    plt.close(fig)

    # --- width ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=150)
    widths, vals = [], []
    for w in ("1", "2", "4", "8"):
        r = next((r for r in rows if r["exp"] == "ksweep" and r["impl"] == "array"
                  and r["array_w"] == w and r["buf_depth"] == "1024"
                  and r["k"] == 1024), None)
        if r:
            widths.append(f"{w}x{w}")
            vals.append(base[1024] / r["cycles"])
    bars = ax.bar(widths, vals, color=["#95a5a6", "#95a5a6", "#2980b9", "#95a5a6"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}x",
                ha="center", fontsize=9)
    ax.set_ylabel("Speedup vs. plain RISC-V (x)")
    ax.set_title("Array width (K = 1024, 1024-word buffer)", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "matmul_width.png")
    fig.savefig(p2)
    plt.close(fig)

    print("wrote", p1)
    print("wrote", p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
