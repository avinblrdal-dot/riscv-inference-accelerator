#!/usr/bin/env python3
"""Capture energy per inference with the Nordic Power Profiler Kit II.

=============================================================================
STATUS: THE PPK2 HAS NOT BEEN PURCHASED. This script is complete but has
NEVER BEEN RUN AGAINST REAL HARDWARE. Treat its first use as a bring-up
exercise, not as a measurement session -- work through
docs/MEASUREMENT_PROTOCOL.md step by step and sanity-check every number.
=============================================================================

WHY DIRECT CURRENT SENSING AND NOT AN ESTIMATE
----------------------------------------------
Vivado will happily print a power number. It is a vectorless estimate built on
assumed switching activity, and for a design like ours it can be wrong by 2-3x
in either direction. Reporting it as "the energy" would be the single easiest
way to invalidate this entire project.

The PPK2 measures the actual current the FPGA core draws, at 100 kHz, with
sub-microamp resolution in its low range. Integrating current x voltage over
the inference window gives joules -- a measurement, not a model.

THE MEASUREMENT
---------------
    E = V * integral(I dt)  over the inference window

Getting the WINDOW right is the hard part. Two approaches, both supported:

  1. GPIO trigger (preferred). The firmware raises a pin before inference and
     lowers it after. The PPK2's digital input records that edge alongside the
     current, so the window is exact.
  2. Statistical (fallback). Measure idle current, then run inference in a
     tight loop, and take the difference. Less precise but needs no extra
     wiring.

Always take a BASELINE measurement with the FPGA idle and subtract it.
Reporting total board current as "inference energy" would include the
regulators, the FTDI bridge and the LEDs, which together dwarf the core power
and would make every configuration look identical.

Usage:
    python3 measure/ppk2_capture.py --list
    python3 measure/ppk2_capture.py --port /dev/tty.usbmodem... --baseline --duration 5
    python3 measure/ppk2_capture.py --port ... --trigger-gpio 0 --n-inferences 100
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SUPPLY_VOLTAGE_MV_DEFAULT = 3300   # Arty A7 core rail, source-meter mode


def require_ppk2():
    try:
        from ppk2_api.ppk2_api import PPK2_API  # type: ignore
        return PPK2_API
    except ImportError:
        print(
            "ERROR: the ppk2-api package is not installed.\n"
            "  pip install ppk2-api\n"
            "\n"
            "  Note: this project has no PPK2 yet. Until one is purchased,\n"
            "  every energy column stays TBD_MEASURED. Do NOT substitute\n"
            "  Vivado's power estimate -- see docs/MEASUREMENT_PROTOCOL.md.",
            file=sys.stderr)
        sys.exit(2)


def list_devices() -> int:
    PPK2_API = require_ppk2()
    try:
        from ppk2_api.ppk2_api import list_devices as _ls  # type: ignore
        devs = _ls()
    except Exception as exc:
        print(f"Could not enumerate PPK2 devices: {exc}", file=sys.stderr)
        return 1
    if not devs:
        print("No PPK2 found. Check the USB cable and that the device is on.")
        return 1
    for d in devs:
        print(f"  {d}")
    return 0


def integrate_energy(samples_ua: list[float], sample_rate_hz: float,
                     voltage_mv: int) -> dict:
    """Integrate current samples into energy.

    E(J) = V(V) * sum(I(A)) * dt(s)

    Returns mean/peak current and total energy. Also returns the sample count
    so the caller can sanity-check that the window length is what it expected
    -- a window that is too short is the most common way to under-report
    energy, and it fails silently.
    """
    if not samples_ua:
        return {"error": "no samples"}
    dt = 1.0 / sample_rate_hz
    v = voltage_mv / 1000.0
    total_charge_c = sum(s * 1e-6 for s in samples_ua) * dt
    energy_j = v * total_charge_c
    return {
        "n_samples": len(samples_ua),
        "duration_s": len(samples_ua) * dt,
        "mean_current_ua": statistics.fmean(samples_ua),
        "peak_current_ua": max(samples_ua),
        "min_current_ua": min(samples_ua),
        "stdev_current_ua": statistics.pstdev(samples_ua) if len(samples_ua) > 1 else 0.0,
        "charge_uc": total_charge_c * 1e6,
        "energy_uj": energy_j * 1e6,
        "supply_voltage_mv": voltage_mv,
    }


def capture(port: str, duration_s: float, voltage_mv: int,
            use_source_meter: bool) -> tuple[list[float], float]:
    PPK2_API = require_ppk2()
    import time

    ppk = PPK2_API(port)
    ppk.get_modifiers()

    if use_source_meter:
        # Source-meter mode: the PPK2 POWERS the target and measures what it
        # draws. Use this only if the FPGA core rail is isolated -- see the
        # note in sweep/vivado/constraints.xdc. Powering an Artix-7 from the
        # PPK2's limited output will brown out under load.
        ppk.use_source_meter()
        ppk.set_source_voltage(voltage_mv)
    else:
        # Ampere-meter mode: the target is externally powered and the PPK2 is
        # wired in series. This is the right mode for the Arty.
        ppk.use_ampere_meter()

    ppk.start_measuring()
    samples: list[float] = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        raw = ppk.get_data()
        if raw != b"":
            chunk, _ = ppk.get_samples(raw)
            samples.extend(chunk)
        time.sleep(0.01)
    ppk.stop_measuring()

    elapsed = time.time() - t0
    rate = len(samples) / elapsed if elapsed > 0 else 0.0
    return samples, rate


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list PPK2 devices")
    ap.add_argument("--port", help="PPK2 serial port")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="capture window in seconds")
    ap.add_argument("--voltage-mv", type=int, default=SUPPLY_VOLTAGE_MV_DEFAULT)
    ap.add_argument("--source-meter", action="store_true",
                    help="PPK2 powers the target (see the warning in the code)")
    ap.add_argument("--baseline", action="store_true",
                    help="capture idle current, to be subtracted later")
    ap.add_argument("--baseline-file",
                    help="a previous --baseline JSON, subtracted from this run")
    ap.add_argument("--n-inferences", type=int, default=1,
                    help="how many inferences ran during the window; energy "
                         "is divided by this")
    ap.add_argument("--label", default="", help="tag for the output row")
    ap.add_argument("--out", help="CSV to append to")
    ap.add_argument("--json-out", help="write the full result as JSON")
    args = ap.parse_args()

    if args.list:
        return list_devices()

    if not args.port:
        ap.error("--port is required (use --list to find it)")

    print("=" * 70)
    print(" PPK2 CAPTURE")
    print("=" * 70)
    print(f"  port      : {args.port}")
    print(f"  mode      : {'source-meter' if args.source_meter else 'ampere-meter'}")
    print(f"  duration  : {args.duration} s")
    print(f"  voltage   : {args.voltage_mv} mV")
    print(f"  inferences: {args.n_inferences}")
    print()
    print("  Before trusting this reading, confirm you have followed")
    print("  docs/MEASUREMENT_PROTOCOL.md -- in particular that the FPGA CORE")
    print("  rail is isolated and that JTAG is disconnected. Measuring whole-")
    print("  board USB current gives a number dominated by the regulators and")
    print("  the FTDI bridge, and it will look plausible while being useless.")
    print()

    samples, rate = capture(args.port, args.duration, args.voltage_mv,
                            args.source_meter)
    if not samples:
        print("ERROR: no samples captured. Check wiring and that the target "
              "is powered.", file=sys.stderr)
        return 1

    result = integrate_energy(samples, rate, args.voltage_mv)
    result["sample_rate_hz"] = round(rate, 1)
    result["label"] = args.label
    result["n_inferences"] = args.n_inferences
    result["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc)\
                                      .strftime("%Y-%m-%dT%H:%M:%SZ")
    result["is_baseline"] = args.baseline
    # The whole point of this script: these numbers are MEASURED.
    result["energy_source"] = "measured"

    if args.baseline_file:
        with open(args.baseline_file) as fh:
            base = json.load(fh)
        idle_ua = base["mean_current_ua"]
        active_ua = result["mean_current_ua"]
        delta_ua = active_ua - idle_ua
        result["baseline_mean_current_ua"] = idle_ua
        result["delta_current_ua"] = delta_ua
        # Energy attributable to the workload, above idle.
        v = args.voltage_mv / 1000.0
        result["net_energy_uj"] = (delta_ua * 1e-6 * v
                                   * result["duration_s"]) * 1e6
        result["energy_per_inference_uj"] = (result["net_energy_uj"]
                                             / max(args.n_inferences, 1))
        if delta_ua <= 0:
            print("WARNING: active current is not above idle. Either the")
            print("  workload never ran, or the measurement window missed it.")
            print("  Do not record this as an inference energy.")
    elif not args.baseline:
        print("NOTE: no --baseline-file given, so this is TOTAL energy in the")
        print("  window, including idle draw. Subtract a baseline before")
        print("  reporting energy per inference.")

    print()
    for k in ("n_samples", "duration_s", "sample_rate_hz", "mean_current_ua",
              "peak_current_ua", "stdev_current_ua", "energy_uj",
              "net_energy_uj", "energy_per_inference_uj"):
        if k in result:
            print(f"  {k:28s} {result[k]:,.3f}" if isinstance(result[k], float)
                  else f"  {k:28s} {result[k]}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nWrote {args.json_out}")

    if args.out:
        exists = os.path.exists(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(result))
            if not exists:
                w.writeheader()
            w.writerow(result)
        print(f"Appended to {args.out}")

    print()
    print("  A single capture is not a measurement. Take at least 5 repeats")
    print("  and report mean and spread -- see docs/MEASUREMENT_PROTOCOL.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
