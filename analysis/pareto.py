#!/usr/bin/env python3
"""Pareto frontier over (energy, accuracy, area) and the minimum-energy pick.

WHY A PARETO FRONTIER RATHER THAN A SINGLE "BEST"
-------------------------------------------------
There is no single best configuration, because the objectives genuinely
conflict: a wider array is faster but larger, and lower precision is smaller
and cheaper but less accurate. Declaring one winner would require secretly
choosing a weighting between joules, percent accuracy and LUTs -- three
quantities with no natural exchange rate.

The Pareto frontier is the honest answer. A configuration is on the frontier
if nothing else beats it on every objective at once. Everything off the
frontier is strictly worse than something on it and can be discarded. What
remains is the real engineering choice, and the paper should present it as a
choice rather than pretending the data picked one.

The one place a single answer IS well posed: "the lowest-energy configuration
that still meets a stated accuracy threshold". That is a constrained
optimisation with an externally given constraint, and --min-accuracy reports it.

ENERGY IS NOT MEASURED YET
--------------------------
Until the PPK2 exists, energy columns are TBD_MEASURED. Rather than refuse to
run, this script falls back to a clearly-labelled PROXY -- cycles x area, a
standard first-order stand-in for energy -- and marks every output as a proxy.
It will not silently pretend the proxy is a measurement.

Usage:
    python3 analysis/pareto.py --csv sweep/results/synthetic_results.csv
    python3 analysis/pareto.py --csv ... --min-accuracy 0.92
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from load_results import load_csv  # noqa: E402


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Average replicates within each configuration, keeping the spread.

    Reporting a single replicate would hide the run-to-run variation in the
    synthesis numbers, which is real and sometimes large for Fmax.
    """
    groups: dict = {}
    for r in rows:
        k = tuple(r.get(x) for x in keys)
        groups.setdefault(k, []).append(r)

    out = []
    for k, members in groups.items():
        agg = dict(zip(keys, k))
        for field in ("cycles_total", "stall_fraction", "utilisation", "luts",
                      "ffs", "dsps", "brams", "fmax_mhz", "accuracy",
                      "energy_per_inference_uj"):
            vals = [m[field] for m in members
                    if isinstance(m.get(field), (int, float))]
            if vals:
                mean = sum(vals) / len(vals)
                agg[field] = mean
                if len(vals) > 1:
                    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                    agg[field + "_sd"] = var ** 0.5
                else:
                    agg[field + "_sd"] = 0.0
        agg["n_replicates"] = len(members)
        agg["accuracy_is_synthetic"] = any(
            m.get("accuracy_is_synthetic") for m in members)
        out.append(agg)
    return out


def energy_proxy(row: dict) -> float | None:
    """First-order energy stand-in: cycles x area.

    Energy is roughly (power x time). Dynamic power scales with the amount of
    switching logic, for which LUT count is a crude proxy; time is the cycle
    count. So cycles x LUTs is a defensible ORDER-OF-MAGNITUDE stand-in.

    It is NOT energy. It ignores static leakage entirely, ignores the
    difference between a DSP slice and a LUT, and ignores memory energy --
    which for this project is probably the dominant term. Every output derived
    from it is labelled PROXY for exactly that reason.
    """
    c = row.get("cycles_total")
    a = row.get("luts")
    if not isinstance(c, (int, float)) or not isinstance(a, (int, float)):
        return None
    return c * a / 1e6


def pareto_front(points: list[dict], objectives: list[tuple[str, str]]
                 ) -> list[dict]:
    """Return the non-dominated subset.

    objectives is a list of (field, direction) where direction is "min" or
    "max". A point is dominated if some other point is at least as good on
    every objective and strictly better on at least one.
    """
    usable = [p for p in points
              if all(isinstance(p.get(f), (int, float)) for f, _ in objectives)]
    front = []
    for p in usable:
        dominated = False
        for q in usable:
            if q is p:
                continue
            at_least_as_good = all(
                (q[f] <= p[f] if d == "min" else q[f] >= p[f])
                for f, d in objectives)
            strictly_better = any(
                (q[f] < p[f] if d == "min" else q[f] > p[f])
                for f, d in objectives)
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def label(p: dict) -> str:
    return (f"{int(p['array_w'])}x{int(p['array_h'])} "
            f"wbuf={int(p['wbuf_depth']):<5d} p={int(p['precision'])}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(ROOT, "sweep", "results",
                                                  "synthetic_results.csv"))
    ap.add_argument("--workload", default=None)
    ap.add_argument("--min-accuracy", type=float, default=None,
                    help="report the lowest-energy config meeting this accuracy")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    if args.workload:
        rows = [r for r in rows if r.get("workload") == args.workload]
    if not rows:
        print("No usable rows.", file=sys.stderr)
        return 1

    workloads = sorted({r["workload"] for r in rows if r.get("workload")})

    for wl in workloads:
        subset = [r for r in rows if r["workload"] == wl]
        aggs = aggregate(subset, ("array_w", "array_h", "wbuf_depth",
                                  "precision", "workload"))

        measured = [a for a in aggs
                    if isinstance(a.get("energy_per_inference_uj"), (int, float))]
        using_proxy = len(measured) == 0
        for a in aggs:
            if using_proxy:
                a["energy"] = energy_proxy(a)
            else:
                a["energy"] = a.get("energy_per_inference_uj")

        print("=" * 84)
        print(f" PARETO FRONTIER -- workload={wl}")
        print("=" * 84)
        if using_proxy:
            print(" ENERGY AXIS IS A PROXY (cycles x LUTs / 1e6), NOT A")
            print(" MEASUREMENT. No PPK2 exists yet, so no energy has been")
            print(" measured. This ordering is indicative only -- it ignores")
            print(" static leakage and memory energy entirely, and memory")
            print(" energy is likely the dominant term. Do NOT put these")
            print(" numbers on the board as energy results.")
        else:
            print(" Energy axis: MEASURED energy per inference (uJ).")
        print()

        front = pareto_front(aggs, [("energy", "min"),
                                    ("accuracy", "max"),
                                    ("luts", "min")])
        front.sort(key=lambda p: p["energy"])

        print(f" {len(front)} of {len(aggs)} configurations are non-dominated")
        print()
        print(f" {'configuration':<26} {'energy':>12} {'accuracy':>9} "
              f"{'LUTs':>7} {'DSPs':>5} {'cycles':>8} {'stall%':>7}")
        print(" " + "-" * 82)
        for p in front:
            print(f" {label(p):<26} {p['energy']:>12.2f} "
                  f"{p.get('accuracy', float('nan')):>9.4f} "
                  f"{p.get('luts', 0):>7.0f} {p.get('dsps', 0):>5.0f} "
                  f"{p.get('cycles_total', 0):>8.0f} "
                  f"{100*p.get('stall_fraction', 0):>6.1f}%")

        if args.min_accuracy is not None:
            print()
            print(f" Minimum-energy configuration with accuracy >= "
                  f"{args.min_accuracy}:")
            ok = [a for a in aggs
                  if isinstance(a.get("accuracy"), (int, float))
                  and a["accuracy"] >= args.min_accuracy
                  and a.get("energy") is not None]
            if not ok:
                print(f"   NONE. No configuration reaches "
                      f"{args.min_accuracy:.3f} accuracy.")
                best = max((a for a in aggs
                            if isinstance(a.get("accuracy"), (int, float))),
                           key=lambda a: a["accuracy"], default=None)
                if best:
                    print(f"   The best available is {best['accuracy']:.4f} "
                          f"at {label(best)}.")
                    print("   Either relax the threshold or improve the model --")
                    print("   this is a model-quality problem, not a hardware one.")
            else:
                best = min(ok, key=lambda a: a["energy"])
                print(f"   {label(best)}")
                print(f"     energy   : {best['energy']:.2f}"
                      f"{'  (PROXY)' if using_proxy else ' uJ (measured)'}")
                print(f"     accuracy : {best['accuracy']:.4f}")
                print(f"     area     : {best.get('luts',0):.0f} LUTs, "
                      f"{best.get('dsps',0):.0f} DSPs")
                print(f"     stalls   : {100*best.get('stall_fraction',0):.1f}% "
                      f"of cycles")

        if any(a.get("accuracy_is_synthetic") for a in aggs):
            print()
            print(" WARNING: accuracy values are tagged SYNTHETIC. They come")
            print(" from a model trained on generated data and are not results.")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
