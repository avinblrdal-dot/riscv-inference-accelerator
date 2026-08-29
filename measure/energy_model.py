#!/usr/bin/env python3
"""System-level node energy MODEL -- projected deployment lifetime.

=============================================================================
EVERYTHING THIS SCRIPT PRINTS IS A MODEL, NOT A MEASUREMENT.
=============================================================================

That distinction is a judging-integrity issue, not a stylistic one. A science
fair judge is entitled to ask "did you measure that or calculate it?", and the
answer must be unambiguous and volunteered rather than extracted. So:

  * every output is prefixed [MODEL]
  * --assumptions dumps every input value with its source
  * inputs that are still guesses are labelled GUESS, and the summary counts
    how many there are
  * the script REFUSES to print a lifetime figure without also printing its
    assumptions

WHAT IT MODELS
--------------
A battery-powered condition-monitoring node spends almost all of its life
asleep. Its energy budget per duty cycle is:

    E_cycle = E_sleep + E_wake + E_sense + E_inference + E_radio

and the deployment lifetime is:

    lifetime = battery_energy / (E_cycle * cycles_per_year)

The interesting consequence, which the sweep exists to quantify: if inference
is a small fraction of E_cycle, then even a 10x faster accelerator barely
moves the lifetime, and the honest conclusion is "optimise the radio instead".
If inference dominates, the accelerator matters enormously. This script is
what turns a cycle count into that answer.

Usage:
    python3 measure/energy_model.py --assumptions
    python3 measure/energy_model.py --inference-energy-uj 45 --duty-cycle-s 60
    python3 measure/energy_model.py --sensitivity
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Assumptions. Each carries a SOURCE tag:
#   MEASURED  -- taken with our own instruments on our own hardware
#   DATASHEET -- from a manufacturer's published figure
#   GUESS     -- an engineering estimate, not yet grounded
# ---------------------------------------------------------------------------
ASSUMPTIONS = {
    "battery_capacity_mah": {
        "value": 2400.0, "unit": "mAh", "source": "DATASHEET",
        "note": "Typical 18650 Li-ion cell",
    },
    "battery_voltage_v": {
        "value": 3.7, "unit": "V", "source": "DATASHEET",
        "note": "Nominal 18650 terminal voltage",
    },
    "battery_derate": {
        "value": 0.80, "unit": "fraction", "source": "GUESS",
        "note": "Usable fraction after self-discharge, temperature and "
                "end-of-life cutoff. Real derating depends on the deployment "
                "environment and has NOT been characterised.",
    },
    "sleep_current_ua": {
        "value": 15.0, "unit": "uA", "source": "GUESS",
        "note": "Deep-sleep current of the whole node. For an FPGA this is "
                "dominated by static leakage and is optimistic -- an Artix-7 "
                "is not a low-power sleep device. A production node would use "
                "an MCU here and power-gate the FPGA entirely.",
    },
    "wake_energy_uj": {
        "value": 120.0, "unit": "uJ", "source": "GUESS",
        "note": "Energy to bring the node from sleep to active, including "
                "regulator settling and, for an FPGA, bitstream reload.",
    },
    "sense_energy_uj": {
        "value": 220.0, "unit": "uJ", "source": "GUESS",
        "note": "I2S MEMS microphone capturing one 1-second window.",
    },
    "inference_energy_uj": {
        "value": None, "unit": "uJ", "source": "TBD_MEASURED",
        "note": "THE NUMBER THIS PROJECT EXISTS TO MEASURE. Requires the "
                "Power Profiler Kit II, which has not been purchased. Supply "
                "it with --inference-energy-uj to explore the model.",
    },
    "radio_energy_uj": {
        "value": 1800.0, "unit": "uJ", "source": "GUESS",
        "note": "One short BLE advertisement carrying a classification "
                "result. Deliberately assumes the node transmits only a "
                "CLASS, not raw audio -- which is the entire argument for "
                "on-device inference.",
    },
    "duty_cycle_s": {
        "value": 60.0, "unit": "s", "source": "GUESS",
        "note": "One measurement per minute. Condition monitoring rarely "
                "needs faster; faults develop over hours or days.",
    },
    "transmit_every_n": {
        "value": 60, "unit": "cycles", "source": "GUESS",
        "note": "Transmit only once an hour unless a fault is detected. This "
                "is where on-device inference pays: without it the node would "
                "have to transmit every window.",
    },
}


def battery_energy_j(a: dict) -> float:
    mah = a["battery_capacity_mah"]["value"]
    v = a["battery_voltage_v"]["value"]
    derate = a["battery_derate"]["value"]
    # mAh -> J:  (mAh / 1000) * 3600 s * V
    return (mah / 1000.0) * 3600.0 * v * derate


def cycle_energy_j(a: dict) -> tuple[float, dict]:
    """Energy for one duty cycle, plus the per-component breakdown."""
    period = a["duty_cycle_s"]["value"]
    sleep_a = a["sleep_current_ua"]["value"] * 1e-6
    v = a["battery_voltage_v"]["value"]

    e_sleep = sleep_a * v * period
    e_wake = a["wake_energy_uj"]["value"] * 1e-6
    e_sense = a["sense_energy_uj"]["value"] * 1e-6
    e_infer = (a["inference_energy_uj"]["value"] or 0.0) * 1e-6
    e_radio = (a["radio_energy_uj"]["value"] * 1e-6
               / max(a["transmit_every_n"]["value"], 1))

    breakdown = {"sleep": e_sleep, "wake": e_wake, "sense": e_sense,
                 "inference": e_infer, "radio": e_radio}
    return sum(breakdown.values()), breakdown


def print_assumptions(a: dict) -> int:
    print("=" * 74)
    print(" MODEL ASSUMPTIONS -- every input, with its provenance")
    print("=" * 74)
    n_guess = 0
    for key, m in a.items():
        val = m["value"]
        shown = "TBD_MEASURED" if val is None else f"{val:g}"
        print(f"  {key:26s} {shown:>14s} {m['unit']:<9s} [{m['source']}]")
        for line in _wrap(m["note"], 66):
            print(f"      {line}")
        if m["source"] == "GUESS":
            n_guess += 1
    print("-" * 74)
    print(f"  {n_guess} of {len(a)} inputs are GUESSES.")
    print("  A lifetime figure is only as good as its weakest assumption.")
    print("  Replace guesses with datasheet or measured values before")
    print("  presenting any projection as meaningful.")
    print("=" * 74)
    return n_guess


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    if cur:
        lines.append(cur)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assumptions", action="store_true",
                    help="dump every input value and exit")
    ap.add_argument("--inference-energy-uj", type=float,
                    help="energy per inference in microjoules")
    ap.add_argument("--duty-cycle-s", type=float)
    ap.add_argument("--battery-mah", type=float)
    ap.add_argument("--sleep-current-ua", type=float)
    ap.add_argument("--sensitivity", action="store_true",
                    help="show how lifetime responds to each input")
    args = ap.parse_args()

    a = {k: dict(v) for k, v in ASSUMPTIONS.items()}

    if args.inference_energy_uj is not None:
        a["inference_energy_uj"]["value"] = args.inference_energy_uj
        a["inference_energy_uj"]["source"] = "USER_SUPPLIED"
        a["inference_energy_uj"]["note"] = (
            "Supplied on the command line. If this did not come from a PPK2 "
            "measurement, the result is a hypothetical, not a projection.")
    if args.duty_cycle_s is not None:
        a["duty_cycle_s"]["value"] = args.duty_cycle_s
    if args.battery_mah is not None:
        a["battery_capacity_mah"]["value"] = args.battery_mah
    if args.sleep_current_ua is not None:
        a["sleep_current_ua"]["value"] = args.sleep_current_ua

    if args.assumptions:
        print_assumptions(a)
        return 0

    if a["inference_energy_uj"]["value"] is None:
        print("=" * 74, file=sys.stderr)
        print(" CANNOT PRODUCE A LIFETIME FIGURE", file=sys.stderr)
        print("=" * 74, file=sys.stderr)
        print(" inference_energy_uj is TBD_MEASURED: the Power Profiler Kit II",
              file=sys.stderr)
        print(" has not been purchased, so this project has no measured value",
              file=sys.stderr)
        print(" for it -- and it will not invent one.", file=sys.stderr)
        print("", file=sys.stderr)
        print(" To explore the model with a hypothetical value:", file=sys.stderr)
        print("   python3 measure/energy_model.py --inference-energy-uj 45",
              file=sys.stderr)
        print("", file=sys.stderr)
        print(" To see every assumption:", file=sys.stderr)
        print("   python3 measure/energy_model.py --assumptions", file=sys.stderr)
        return 1

    # The model refuses to print a number without its assumptions attached.
    n_guess = print_assumptions(a)
    print()

    e_batt = battery_energy_j(a)
    e_cycle, parts = cycle_energy_j(a)
    period = a["duty_cycle_s"]["value"]
    cycles_per_year = 365.25 * 24 * 3600 / period
    lifetime_years = e_batt / (e_cycle * cycles_per_year)

    print("=" * 74)
    print(" [MODEL] PROJECTED NODE LIFETIME -- CALCULATED, NOT MEASURED")
    print("=" * 74)
    print(f"  usable battery energy : {e_batt:,.0f} J")
    print(f"  duty cycle            : every {period:g} s "
          f"({cycles_per_year:,.0f} cycles/year)")
    print()
    print("  Energy per duty cycle:")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        share = 100.0 * v / e_cycle if e_cycle else 0.0
        bar = "#" * int(share / 2)
        print(f"    {k:10s} {v*1e6:9.1f} uJ  {share:5.1f}%  {bar}")
    print(f"    {'TOTAL':10s} {e_cycle*1e6:9.1f} uJ")
    print()
    print(f"  [MODEL] projected lifetime: {lifetime_years:.2f} years")
    print()

    # The interpretation that actually matters for the research question.
    infer_share = 100.0 * parts["inference"] / e_cycle if e_cycle else 0.0
    print("  Interpretation:")
    if infer_share < 5:
        print(f"    Inference is only {infer_share:.1f}% of the cycle energy.")
        print("    Even a 10x faster accelerator would change the lifetime")
        print("    very little. The binding constraint is elsewhere -- look at")
        print(f"    {max(parts, key=parts.get)}. This is an honest and")
        print("    publishable negative result about system-level design, and")
        print("    it is worth reporting rather than hiding.")
    elif infer_share > 40:
        print(f"    Inference is {infer_share:.1f}% of the cycle energy and")
        print("    dominates the budget. Accelerating it translates almost")
        print("    directly into deployment lifetime.")
    else:
        print(f"    Inference is {infer_share:.1f}% of the cycle energy.")
        print("    Amdahl's law at the SYSTEM level: the best possible")
        print(f"    lifetime improvement from a perfect accelerator is")
        print(f"    {1/(1-infer_share/100):.2f}x.")
    print()
    print(f"  REMINDER: this is a MODEL built on {n_guess} guessed inputs.")
    print("  It is not a measurement and must be labelled as a model wherever")
    print("  it appears -- in the paper, on the board, and in conversation.")
    print("=" * 74)

    if args.sensitivity:
        print()
        print(" [MODEL] SENSITIVITY -- lifetime vs each input, +/- 50%")
        print("-" * 74)
        base = lifetime_years
        for key in ("inference_energy_uj", "sleep_current_ua",
                    "radio_energy_uj", "sense_energy_uj", "duty_cycle_s"):
            row = []
            for mult in (0.5, 2.0):
                b = {k: dict(v) for k, v in a.items()}
                b[key]["value"] = a[key]["value"] * mult
                ec, _ = cycle_energy_j(b)
                cpy = 365.25*24*3600 / b["duty_cycle_s"]["value"]
                row.append(battery_energy_j(b) / (ec * cpy))
            print(f"  {key:24s} half: {row[0]:6.2f} y   "
                  f"base: {base:6.2f} y   double: {row[1]:6.2f} y")
        print("-" * 74)
        print("  The input with the widest spread is the one worth measuring")
        print("  first. If that is not inference energy, say so in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
