#!/usr/bin/env python3
"""Matrix-multiply benchmark sweep: at what data size does the speedup stop?

Runs sw/bench/matmul.c on the real SoC under Verilator:

  * the CPU image (software baseline + DOT4) ONCE -- it never touches the MAC
    array, so the array's shape cannot change its cycle count;
  * the ARRAY image on every array width x buffer depth in
    sweep/sweep_config.yaml (16 hardware builds).

Every measurement's checksum is checked against C = A x B recomputed here in
Python from the same xorshift32 operands. A row whose checksum does not match
is marked result_correct=False and excluded from the summary -- the same
"fast and wrong is never fast" rule as sweep/run_sweep.py.

Outputs:
  sweep/results/matmul_results.csv   every measurement (regenerable, not in git)
  sweep/results/MATMUL_SUMMARY.md    the tables, in plain language (tracked)

Usage:
  make -C sw matmul BUILD=build_matmul          # build the two firmware images
  python3 sweep/run_matmul_sweep.py             # full sweep
  python3 sweep/run_matmul_sweep.py --widths 4 --depths 256   # one config
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from run_sweep import build_verilator, git_sha  # noqa: E402

FW_DIR = os.path.join(ROOT, "sw", "build_matmul")
# Verilator object directories live OUTSIDE the repo: GNU Make refuses to
# build in any directory whose real path contains a space, and this repo sits
# under ".../Science Fair/...". The symlink trick in run_sweep.space_free_root
# fixes the source paths but not the build directory, because make resolves
# the real path. The system temp dir has no spaces, and builds are cached
# there between runs (each hardware config takes ~1 minute to compile).
import tempfile  # noqa: E402
OBJ_ROOT = os.path.join(tempfile.gettempdir(), "riscv_matmul_obj")
CSV_OUT = os.path.join(ROOT, "sweep", "results", "matmul_results.csv")
MD_OUT = os.path.join(ROOT, "sweep", "results", "MATMUL_SUMMARY.md")

# Must match sw/bench/matmul.c.
MATMUL_SEED = 0x2545F491
MAX_OPERAND_BYTES = 16384

CSV_COLUMNS = [
    "exp", "impl", "array_w", "array_h", "buf_depth", "m", "n", "k",
    "mode", "chunk", "cycles", "macs", "cycles_per_mac", "checksum",
    "expected_checksum", "result_correct", "timeouts", "git_sha",
    "timestamp_utc",
]


# ---------------------------------------------------------------------------
# Python reference -- bit-exact twin of the firmware's data and checksum
# ---------------------------------------------------------------------------
def operands() -> tuple[np.ndarray, np.ndarray]:
    x = MATMUL_SEED
    out = np.empty(2 * MAX_OPERAND_BYTES, dtype=np.int8)
    for i in range(2 * MAX_OPERAND_BYTES):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = np.int8(np.uint8(x >> 24).view(np.int8))
    return out[:MAX_OPERAND_BYTES], out[MAX_OPERAND_BYTES:]


def expected_checksum(a_flat, bt_flat, m: int, n: int, k: int) -> int:
    a = a_flat[: m * k].astype(np.int64).reshape(m, k)
    bt = bt_flat[: n * k].astype(np.int64).reshape(n, k)
    c = (a @ bt.T).astype(np.int64).ravel()
    h = 0
    for v in c:
        h = (h * 31 + (int(v) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h   # firmware prints int32


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_image(hex_name: str, array_w: int, depth: int, timeout: int) -> tuple[list[dict], str]:
    objdir = os.path.join(OBJ_ROOT, f"w{array_w}_d{depth}")
    os.makedirs(objdir, exist_ok=True)
    exe = os.path.join(objdir, "Vsoc_top")
    if not os.path.exists(exe):
        ok, err = build_verilator(array_w, array_w, depth, depth, 8, objdir)
        if not ok:
            return [], "verilator build failed: " + err
    fw = os.path.join(FW_DIR, hex_name)
    dst = os.path.join(ROOT, "sim", "build", "firmware.hex")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(fw, dst)
    try:
        res = subprocess.run([exe, "+max_cycles=4000000000", "+progress=0"],
                             capture_output=True, text=True, timeout=timeout,
                             cwd=ROOT)
    except subprocess.TimeoutExpired:
        return [], "wall-clock timeout"
    text = res.stdout + res.stderr
    if "=== done ===" not in text:
        return [], "firmware did not finish:\n" + text[-600:]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("mm "):
            continue
        rec = dict(tok.split("=", 1) for tok in line.split()[1:])
        rows.append(rec)
    return rows, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--depths", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    for h in ("matmul_cpu.hex", "matmul_array.hex"):
        if not os.path.exists(os.path.join(FW_DIR, h)):
            print(f"ERROR: sw/build_matmul/{h} missing -- run "
                  f"'make -C sw matmul BUILD=build_matmul' first.", file=sys.stderr)
            return 1

    a_flat, bt_flat = operands()
    expect_cache: dict = {}

    def expect(m, n, k):
        key = (m, n, k)
        if key not in expect_cache:
            expect_cache[key] = expected_checksum(a_flat, bt_flat, m, n, k)
        return expect_cache[key]

    sha = git_sha()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_rows: list[dict] = []
    n_wrong = 0

    def record(rec: dict, aw, depth):
        nonlocal n_wrong
        m, n, k = int(rec["m"]), int(rec["n"]), int(rec["k"])
        cyc = int(rec["cycles"])
        got = int(rec["checksum"])
        exp = expect(m, n, k)
        ok = (got == exp) and int(rec.get("timeouts", 0)) == 0
        if not ok:
            n_wrong += 1
        macs = m * n * k
        out_rows.append({
            "exp": rec["exp"], "impl": rec["impl"],
            "array_w": aw, "array_h": aw, "buf_depth": depth,
            "m": m, "n": n, "k": k,
            "mode": rec.get("mode", ""), "chunk": rec.get("chunk", ""),
            "cycles": cyc, "macs": macs,
            "cycles_per_mac": round(cyc / macs, 3),
            "checksum": got, "expected_checksum": exp,
            "result_correct": ok, "timeouts": rec.get("timeouts", ""),
            "git_sha": sha, "timestamp_utc": stamp,
        })
        return ok

    # --- CPU image, once --------------------------------------------------
    print("CPU image (baseline + DOT4), run once ...", flush=True)
    rows, err = run_image("matmul_cpu.hex", 1, 16, args.timeout)
    if err:
        print("  FAIL:", err)
        return 1
    for rec in rows:
        ok = record(rec, "", "")
        print(f"  {rec['exp']:7s} {rec['impl']:8s} m={rec['m']:>3} n={rec['n']:>3} "
              f"k={rec['k']:>5} cycles={int(rec['cycles']):>11,} {'OK' if ok else 'WRONG'}")

    # --- ARRAY image, every config ------------------------------------------
    for aw in args.widths:
        for depth in args.depths:
            print(f"ARRAY {aw}x{aw} buffer={depth} ...", flush=True)
            rows, err = run_image("matmul_array.hex", aw, depth, args.timeout)
            if err:
                print("  FAIL:", err)
                continue
            for rec in rows:
                ok = record(rec, aw, depth)
                print(f"  {rec['exp']:7s} k={rec['k']:>5} n={rec['n']:>3} "
                      f"{rec['mode']:8s} chunk={rec['chunk']:>4} "
                      f"cycles={int(rec['cycles']):>11,} {'OK' if ok else 'WRONG'}")

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {CSV_OUT}: {len(out_rows)} rows, {n_wrong} wrong")

    write_summary(out_rows)
    print(f"Wrote {MD_OUT}")
    return 0 if n_wrong == 0 else 2


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def write_summary(rows: list[dict]) -> None:
    good = [r for r in rows if r["result_correct"]]

    def cpu(exp, impl, m, n, k):
        for r in good:
            if (r["exp"], r["impl"], r["m"], r["n"], r["k"]) == (exp, impl, m, n, k):
                return r["cycles"]
        return None

    configs = sorted({(r["array_w"], r["buf_depth"]) for r in good if r["impl"] == "array"})
    L = []
    L.append("# Matrix-multiply benchmark: results")
    L.append("")
    L.append(f"Generated by `sweep/run_matmul_sweep.py` at commit `{rows[0]['git_sha'] if rows else '?'}`. "
             "All numbers are simulated clock cycles on the real SoC (PicoRV32 + accelerator) "
             "under Verilator. Every number in these tables passed the checksum check against "
             f"Python. Rows that failed: {len(rows) - len(good)}.")
    L.append("")
    L.append("Speedup = baseline cycles / accelerated cycles. The baseline is plain rv32i C "
             "(every multiply is a software routine). Higher is better.")
    L.append("")

    # ksweep
    ks = sorted({r["k"] for r in good if r["exp"] == "ksweep"})
    L.append("## 1. Speedup vs. dot-product length K (16x16 output)")
    L.append("")
    L.append("This is the \"how big a chunk of data\" question. Each output is a dot product "
             "of length K. `*` marks configurations where K was larger than the buffer, so the "
             "data had to be split into pieces and the weights could not stay loaded.")
    L.append("")
    hdr = "| K | baseline cycles | DOT4 | " + " | ".join(f"{w}x{w}/{d}" for w, d in configs) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (3 + len(configs)))
    for k in ks:
        base = cpu("ksweep", "baseline", 16, 16, k)
        d4 = cpu("ksweep", "dot4", 16, 16, k)
        cells = [str(k), f"{base:,}" if base else "-",
                 f"{base / d4:.2f}x" if base and d4 else "-"]
        for w, d in configs:
            r = next((r for r in good if r["exp"] == "ksweep" and r["impl"] == "array"
                      and r["array_w"] == w and r["buf_depth"] == d and r["k"] == k), None)
            if r and base:
                cells.append(f"{base / r['cycles']:.2f}x" + ("*" if r["mode"] == "chunked" else ""))
            else:
                cells.append("-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # chunk
    L.append("## 2. Cost of splitting the data into chunks (16x16 output, K = 256)")
    L.append("")
    L.append("Here the weights are deliberately NOT kept loaded: every pass sends a chunk "
             "of weights and a chunk of inputs, runs, and adds the partial sums in software. "
             "Shows how much the chunk size alone matters.")
    L.append("")
    chunks = sorted({int(r["chunk"]) for r in good if r["exp"] == "chunk"})
    base = cpu("ksweep", "baseline", 16, 16, 256)
    L.append("| chunk | " + " | ".join(f"{w}x{w}/{d}" for w, d in configs) + " |")
    L.append("|" + "---|" * (1 + len(configs)))
    for ch in chunks:
        cells = [str(ch)]
        for w, d in configs:
            r = next((r for r in good if r["exp"] == "chunk" and r["array_w"] == w
                      and r["buf_depth"] == d and int(r["chunk"]) == ch), None)
            cells.append(f"{base / r['cycles']:.2f}x" if r and base else "-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # square
    L.append("## 3. Speedup vs. matrix size (square N x N times N x N)")
    L.append("")
    ns = sorted({r["n"] for r in good if r["exp"] == "square"})
    L.append("| N | baseline cycles | DOT4 | " + " | ".join(f"{w}x{w}/{d}" for w, d in configs) + " |")
    L.append("|" + "---|" * (3 + len(configs)))
    for n in ns:
        base = cpu("square", "baseline", n, n, n)
        d4 = cpu("square", "dot4", n, n, n)
        cells = [str(n), f"{base:,}" if base else "-",
                 f"{base / d4:.2f}x" if base and d4 else "-"]
        for w, d in configs:
            r = next((r for r in good if r["exp"] == "square" and r["impl"] == "array"
                      and r["array_w"] == w and r["buf_depth"] == d and r["n"] == n), None)
            if r and base:
                cells.append(f"{base / r['cycles']:.2f}x" + ("*" if r["mode"] == "chunked" else ""))
            else:
                cells.append("-")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("Column labels are `array width x height / buffer depth (words)`.")
    L.append("")

    with open(MD_OUT, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
