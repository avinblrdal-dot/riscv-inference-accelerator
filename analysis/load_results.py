#!/usr/bin/env python3
"""Load sweep results, enforce the integrity rules, and synthesise test data.

TWO JOBS
--------
1. Load sweep/results/*.csv into a tidy table, DROPPING invalid rows loudly
   rather than silently averaging failures in as zeros.
2. Generate synthetic results with a KNOWN ground-truth effect structure, so
   analysis/anova.py and analysis/pareto.py can be developed and validated
   before any real sweep has run.

Point 2 matters more than it sounds. An ANOVA script that has never been run
on data with a known answer is a script nobody should trust. The synthetic
generator below injects a deliberate array_w x wbuf_depth INTERACTION, so we
can confirm the analysis actually detects the thing RQ3 is about -- rather
than discovering on the night before the deadline that the model was
mis-specified.

Uses pandas when available and falls back to a plain-dict table otherwise, so
loading never becomes the reason someone cannot reproduce a figure.

Usage:
    python3 analysis/load_results.py --csv sweep/results/sweep_results.csv
    python3 analysis/load_results.py --synthetic --out sweep/results/synthetic.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NUMERIC = {
    "array_w", "array_h", "wbuf_depth", "abuf_depth", "precision", "replicate",
    "cycles_total", "cycles_active", "cycles_stall", "macs_done",
    "utilisation", "stall_fraction", "luts", "ffs", "dsps", "brams",
    "fmax_mhz", "wns_ns", "accuracy", "energy_per_inference_uj",
}
BOOLEAN = {"sim_ok", "synth_ok", "timing_met", "accuracy_is_synthetic"}


def _coerce_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if v is None or v == "":
            out[k] = None
        elif k in BOOLEAN:
            out[k] = str(v).strip().lower() in ("true", "1", "yes")
        elif k in NUMERIC:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = None
        else:
            out[k] = v
    return out


def load_csv(path: str, require_synthesis: bool = False,
             verbose: bool = True) -> list[dict]:
    """Load one results CSV, applying the integrity rules from SCHEMA.md."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"  Generate real results with:  python3 sweep/run_sweep.py --dry-run\n"
            f"  Or synthetic ones with:      python3 analysis/load_results.py --synthetic"
        )

    with open(path, newline="") as fh:
        rows = [_coerce_row(r) for r in csv.DictReader(fh)]

    n_total = len(rows)
    kept, dropped = [], []
    for r in rows:
        # Rule 1 from sweep/results/SCHEMA.md: a failed run is NOT a zero.
        if r.get("sim_ok") is False:
            dropped.append((r, "sim_ok=False"))
            continue
        if require_synthesis and r.get("synth_ok") is False:
            dropped.append((r, "synth_ok=False"))
            continue
        kept.append(r)

    if verbose:
        print(f"Loaded {path}")
        print(f"  {n_total} rows, {len(kept)} usable, {len(dropped)} dropped")
        if dropped:
            print("  Dropped (excluded from analysis, NOT treated as zeros):")
            reasons: dict[str, int] = {}
            for _, why in dropped:
                reasons[why] = reasons.get(why, 0) + 1
            for why, n in sorted(reasons.items()):
                print(f"    {why}: {n}")

        # Loud, unmissable warnings about provenance.
        synth_acc = [r for r in kept if r.get("accuracy_is_synthetic")]
        if synth_acc:
            print(f"  WARNING: {len(synth_acc)} rows have "
                  f"accuracy_is_synthetic=True.")
            print("           Those accuracies come from a model trained on")
            print("           synthetic data and are NOT results.")

        srcs = {r.get("energy_source") for r in kept}
        if srcs and srcs <= {"TBD_MEASURED", None}:
            print("  NOTE: every energy column is TBD_MEASURED -- no PPK2 yet.")
            print("        Any energy figure produced downstream is a MODEL.")
    return kept


def to_dataframe(rows: list[dict]):
    """Return a pandas DataFrame if pandas is installed, else the list."""
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def synthesize_results(seed: int = 20260828, replicates: int = 5) -> list[dict]:
    """Generate results with a KNOWN effect structure, for testing the analysis.

    The ground truth injected here, so the analysis can be validated against
    something whose answer we already know:

      * a main effect of array_w      (wider array -> fewer cycles)
      * a main effect of wbuf_depth   (buffering -> fewer stalls)
      * a STRONG array_w x wbuf_depth INTERACTION: buffering helps a lot at
        width 8 and almost not at all at width 1, because a 1-wide array is
        never bandwidth-starved in the first place
      * a main effect of precision on area (int4 is smaller)
      * realistic run-to-run noise on the synthesis numbers only

    If analysis/anova.py cannot recover that interaction from this data, the
    ANOVA model is mis-specified and would also miss it in the real sweep.
    """
    rng = random.Random(seed)
    rows = []

    for aw in (1, 2, 4, 8):
        for wbuf in (0, 64, 256, 1024):
            for prec in (8, 4):
                for wl in ("workload_a", "workload_b"):
                    for rep in range(replicates):
                        ah = aw
                        # Operand bytes per step vs a 4-byte bus.
                        need = (aw + ah) / (2 if prec == 4 else 1)
                        beats = 1 if wbuf > 0 else max(1, math.ceil(need / 4))

                        base_k = 256 if wl == "workload_a" else 8
                        n_tiles = max(1, (4 if wl == "workload_a" else 128) // aw)
                        active = base_k * n_tiles
                        stalls = active * (beats - 1)
                        # Partial buffers help partially -- a small buffer
                        # cannot hold a whole tile, so some refetching remains.
                        if 0 < wbuf < 256:
                            stalls += int(active * 0.15)
                        cycles = active + stalls + 4 * n_tiles

                        macs = active * aw * ah
                        util = macs / max(aw * ah * cycles, 1)

                        # Area: multipliers scale with cells, and int4 costs
                        # roughly half the LUTs per multiplier.
                        cells = aw * ah
                        lut_per_cell = 62 if prec == 8 else 34
                        luts = int(1850 + cells * lut_per_cell
                                   + (wbuf / 64.0) * 22 + rng.gauss(0, 40))
                        ffs = int(1200 + cells * 38 + rng.gauss(0, 25))
                        dsps = cells if prec == 8 else max(1, cells // 2)
                        brams = round(max(0.0, wbuf / 512.0) * 2, 1)

                        # Fmax falls with width (longer broadcast nets) and is
                        # genuinely noisy between placer seeds.
                        fmax = 128.0 - 5.2 * math.log2(max(aw, 1)) ** 2 \
                            + rng.gauss(0, 3.5)
                        wns = 10.0 - 1000.0 / max(fmax, 1)

                        # Accuracy: int4 costs a little.
                        acc = (0.94 if prec == 8 else 0.906) + rng.gauss(0, 0.004)

                        rows.append({
                            "array_w": aw, "array_h": ah,
                            "wbuf_depth": wbuf, "abuf_depth": wbuf,
                            "precision": prec, "workload": wl,
                            "replicate": rep,
                            "cycles_total": cycles, "cycles_active": active,
                            "cycles_stall": stalls, "macs_done": macs,
                            "utilisation": round(util, 6),
                            "stall_fraction": round(stalls / max(cycles, 1), 6),
                            "sim_ok": True,
                            "luts": luts, "ffs": ffs, "dsps": dsps,
                            "brams": brams,
                            "fmax_mhz": round(fmax, 2),
                            "wns_ns": round(wns, 4),
                            "timing_met": wns >= 0, "synth_ok": True,
                            "accuracy": round(min(acc, 0.999), 5),
                            "accuracy_is_synthetic": True,
                            "energy_per_inference_uj": "TBD_MEASURED",
                            "energy_source": "TBD_MEASURED",
                            "git_sha": "synthetic",
                            "timestamp_utc": "synthetic",
                            "iverilog_version": "", "vivado_version": "",
                            "config_sha256": "synthetic", "seed": seed + rep,
                        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(ROOT, "sweep", "results",
                                                  "sweep_results.csv"))
    ap.add_argument("--synthetic", action="store_true",
                    help="generate test data with a known effect structure")
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", help="write the generated data here")
    args = ap.parse_args()

    if args.synthetic:
        rows = synthesize_results(args.seed, args.replicates)
        out = args.out or os.path.join(ROOT, "sweep", "results",
                                       "synthetic_results.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out}: {len(rows)} synthetic rows")
        print()
        print("  SYNTHETIC DATA with a deliberately injected")
        print("  array_w x wbuf_depth INTERACTION. It exists to validate")
        print("  analysis/anova.py against a known answer. It is NOT a result")
        print("  and must never appear in the paper or on the board.")
        return 0

    rows = load_csv(args.csv)
    print()
    print(f"  {len(rows)} usable rows")
    if rows:
        keys = ("array_w", "wbuf_depth", "precision", "workload")
        for k in keys:
            vals = sorted({r[k] for r in rows if r.get(k) is not None})
            print(f"  {k:12s}: {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
