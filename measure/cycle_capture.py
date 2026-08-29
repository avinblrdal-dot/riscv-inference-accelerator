#!/usr/bin/env python3
"""Parse cycle counts from the firmware's UART output.

The firmware (sw/src/main.c) prints one `key=value` per line. This turns that
into a tidy CSV row that joins onto the sweep results.

Works from a live serial port OR from a captured text file, which means the
parser can be developed and tested with no board attached -- and means a
recorded log can be re-parsed later if the analysis changes.

Usage:
    python3 measure/cycle_capture.py --port /dev/tty.usbserial-210319B --out run.csv
    python3 measure/cycle_capture.py --file sim_output.txt --out run.csv
    python3 measure/cycle_capture.py --file - < sim_output.txt
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

KV = re.compile(r"^(\w+)=(-?\d+)\s*$")
SECTION = re.compile(r"^---\s*(\w+)\s*---\s*$")


def parse_stream(lines) -> tuple[list[dict], dict]:
    """Return (per-variant records, run metadata)."""
    records, meta, current = [], {}, None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = SECTION.match(line)
        if m:
            if current:
                records.append(current)
            current = {"variant": m.group(1)}
            continue

        if line.startswith("model="):
            meta["model"] = line.split("=", 1)[1]
            continue
        if line.startswith("hash="):
            meta["model_hash"] = line.split("=", 1)[1]
            continue

        m = KV.match(line)
        if m:
            key, val = m.group(1), int(m.group(2))
            if current is None:
                meta[key] = val
            else:
                current[key] = val

    if current:
        records.append(current)
    return records, meta


def derive(rec: dict) -> dict:
    """Add the derived quantities the analysis needs.

    mac_fraction is the Amdahl-relevant number: the share of total cycles that
    an accelerator could in principle address. The firmware reports it in
    permille to avoid needing floating point on a core with no FPU, so we
    divide by 1000 here rather than recomputing it.
    """
    out = dict(rec)
    total = rec.get("cycles_total", 0)
    mac = rec.get("cycles_mac", 0)

    if total:
        out["mac_fraction"] = round(mac / total, 6)
        # Amdahl's ceiling: the best possible overall speedup if the MAC
        # portion were made infinitely fast.
        frac = mac / total
        out["amdahl_max_speedup"] = round(1.0 / (1.0 - frac), 4) if frac < 1 else float("inf")
        if rec.get("instret"):
            out["ipc"] = round(rec["instret"] / total, 4)

    accel_cycles = rec.get("accel_cycles", 0)
    if accel_cycles:
        out["accel_stall_fraction"] = round(
            rec.get("accel_stalls", 0) / accel_cycles, 6)

    # A nonzero timeout count means the accelerator wedged; the run's numbers
    # are not trustworthy and must not be silently averaged in.
    if rec.get("accel_timeouts", 0):
        out["valid"] = False
        out["invalid_reason"] = f"{rec['accel_timeouts']} accelerator timeouts"
    else:
        out["valid"] = True
        out["invalid_reason"] = ""
    return out


def read_serial(port: str, baud: int, timeout_s: float) -> list[str]:
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial is not installed.\n"
              "  pip install pyserial\n"
              "  Or capture the output to a file and use --file instead.",
              file=sys.stderr)
        sys.exit(2)

    print(f"Opening {port} at {baud} baud...")
    print("  Waiting for '=== done ===' from the firmware.")
    print("  Press the board's reset button if nothing appears.")
    lines: list[str] = []
    with serial.Serial(port, baud, timeout=1) as ser:
        import time
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace").rstrip()
            print(f"  | {text}")
            lines.append(text)
            if "=== done ===" in text:
                break
        else:
            print(f"WARNING: timed out after {timeout_s}s without seeing "
                  f"'=== done ==='. The capture may be incomplete.",
                  file=sys.stderr)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="serial device, e.g. /dev/tty.usbserial-*")
    src.add_argument("--file", help="a captured text file, or - for stdin")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", help="CSV output path")
    ap.add_argument("--config-name", default="", help="workload tag for the row")
    args = ap.parse_args()

    if args.port:
        lines = read_serial(args.port, args.baud, args.timeout)
        source = args.port
    elif args.file == "-":
        lines = sys.stdin.read().splitlines()
        source = "stdin"
    else:
        with open(args.file) as fh:
            lines = fh.read().splitlines()
        source = args.file

    records, meta = parse_stream(lines)
    if not records:
        print("ERROR: no variant sections found. Expected lines like "
              "'--- baseline ---' followed by 'cycles_total=NNN'.",
              file=sys.stderr)
        return 1

    stamp = datetime.datetime.now(datetime.timezone.utc)\
                    .strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for rec in records:
        row = derive(rec)
        row.update({
            "source": source,
            "timestamp_utc": stamp,
            "model": meta.get("model", ""),
            "model_hash": meta.get("model_hash", ""),
            "config_name": args.config_name,
            # This script reads CYCLES, never energy. Energy comes only from
            # the PPK2 (measure/ppk2_capture.py). Saying so on every row makes
            # it impossible to mix the two up later.
            "energy_source": "not_measured_here",
        })
        rows.append(row)

    print()
    print(f"{'variant':<10} {'cycles':>10} {'mac cycles':>11} "
          f"{'mac %':>7} {'amdahl':>8} {'IPC':>6}")
    print("-" * 60)
    for r in rows:
        print(f"{r.get('variant',''):<10} {r.get('cycles_total',0):>10} "
              f"{r.get('cycles_mac',0):>11} "
              f"{100*r.get('mac_fraction',0):>6.1f}% "
              f"{r.get('amdahl_max_speedup',0):>7.2f}x "
              f"{r.get('ipc',0):>6.2f}")

    base = next((r for r in rows if r.get("variant") == "baseline"), None)
    if base and base.get("cycles_total"):
        print()
        print("Speedup vs baseline (cycles):")
        for r in rows:
            if r is base:
                continue
            sp = base["cycles_total"] / max(r.get("cycles_total", 1), 1)
            print(f"  {r.get('variant',''):<10} {sp:.2f}x")
        print()
        print(f"  Amdahl ceiling from the baseline's MAC fraction: "
              f"{base.get('amdahl_max_speedup', 0):.2f}x")
        print("  Any measured speedup above that ceiling means the variants")
        print("  are not doing the same work -- check parity before believing it.")

    invalid = [r for r in rows if not r.get("valid", True)]
    if invalid:
        print()
        print("WARNING: some runs are INVALID and must be excluded:")
        for r in invalid:
            print(f"  {r.get('variant')}: {r.get('invalid_reason')}")

    if args.out:
        cols = sorted({k for r in rows for k in r})
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
