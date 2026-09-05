#!/usr/bin/env python3
"""Build, simulate and synthesise every configuration in the design space.

WHAT IT DOES
------------
For each cell of the full factorial in sweep/sweep_config.yaml:
  1. builds the RTL with those parameters (via iverilog -P overrides)
  2. runs a simulation to get cycle, stall and MAC counts
  3. optionally runs Vivado synthesis for LUT/FF/DSP/BRAM and timing
  4. records accuracy from the corresponding quantized model
  5. writes one tidy CSV row per (config, workload, replicate)

--dry-run skips Vivado, so the entire pipeline can be developed and tested
before anyone has Vivado installed or a board on the desk. That is the normal
mode until the hardware arrives.

ON REPLICATES
-------------
RTL simulation is deterministic: the same parameters give the same cycle
counts every time, so replicating the simulation measures nothing. Replicates
exist for the SYNTHESIS numbers, which are not deterministic -- Vivado's
placer and router are seeded and heuristic, so LUT counts and especially Fmax
vary between runs of the identical design.

Reporting a single Fmax as if it were exact would be wrong. run_sweep.py
therefore runs synthesis `replicates` times with different seeds and the
analysis reports mean and spread. In --dry-run mode replicates are collapsed
to 1 with a note, because duplicating deterministic rows would fake a
precision that does not exist.

Usage:
    python3 sweep/run_sweep.py --dry-run              # simulation only
    python3 sweep/run_sweep.py --dry-run --quick      # a small subset
    python3 sweep/run_sweep.py                        # full, needs Vivado
    python3 sweep/run_sweep.py --filter array_w=8     # one slice
"""

from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

from config import load_config          # noqa: E402
from freeze import git_sha, config_hash  # noqa: E402

VARIANT = os.environ.get("SWEEP_VARIANT", "array")

RTL_FILES = [
    "dot4_pcpi.v", "mac_unit.v", "mac_array.v", "weight_buffer.v",
    "activation_buffer.v", "accel_ctrl.v", "accel_top.v", "perf_counter.v",
    "requantize.v", "uart_tx.v", "soc_top.v",
]

CSV_COLUMNS = [
    "array_w", "array_h", "wbuf_depth", "abuf_depth", "precision",
    "workload", "replicate",
    "cycles_total", "cycles_active", "cycles_stall", "macs_done",
    "accel_cycles", "utilisation", "accel_busy_fraction", "stall_fraction",
    "sim_ok", "predicted_class", "expected_class", "result_correct",
    "accel_timeouts",
    "luts", "ffs", "dsps", "brams", "fmax_mhz", "wns_ns",
    "timing_met", "synth_ok",
    "accuracy", "accuracy_is_synthetic",
    "energy_per_inference_uj", "energy_source",
    "git_sha", "timestamp_utc", "iverilog_version", "vivado_version",
    "config_sha256", "seed",
]

# ---------------------------------------------------------------------------
# Simulation backend
# ---------------------------------------------------------------------------
# This runs the REAL firmware on the REAL SoC under Verilator, not a
# standalone FSM. An earlier version simulated accel_ctrl.v in isolation with
# synthetic dimensions, which measured the controller's arithmetic but could
# not observe the thing that actually dominates: the CPU packing operands and
# pushing them over the bus. It also could not tell whether a configuration
# computed the CORRECT ANSWER.
#
# That last point is not optional. A configuration that is fast and wrong must
# never be recorded as fast -- this project has already produced one such
# result (docs/REVIEW.md, addendum 2), and the only reason it was caught is
# that something checked the output against a golden reference.

def space_free_root() -> str:
    """A path to this repo containing no whitespace, for Verilator's --build.

    Verilator's generated Vsoc_top.mk lists source files space-separated and
    unquoted. If ROOT itself contains a space (this repo currently lives
    under ".../Science Fair/riscv-inference-accelerator" -- it did not when
    the sweep was first built, and moving it is outside this script's
    control), that source list silently splits into two bogus Make targets
    and every configuration in the sweep fails with a "No rule to make
    target" error whose target is half a real path. iverilog and gcc are
    unaffected -- this is specific to verilator's own generated Makefile, not
    to Python's subprocess handling (which passes argv correctly either way).

    Rather than require a specific directory name, work around it with a
    symlink at a fixed, space-free location and build through that instead.
    The symlink is recreated if it ever points somewhere stale.
    """
    if " " not in ROOT:
        return ROOT
    link = "/tmp/riscv_inference_accelerator_root"
    target = os.path.realpath(ROOT)
    if os.path.islink(link):
        if os.path.realpath(link) != target:
            os.remove(link)
            os.symlink(target, link)
    elif os.path.exists(link):
        raise RuntimeError(
            f"{link} exists and is not the symlink the Verilator build "
            f"needs -- remove it manually and re-run.")
    else:
        os.symlink(target, link)
    return link


def tool_version(cmd: list[str], pattern: str = r"([\d.]+)") -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        text = (out.stdout or "") + (out.stderr or "")
        m = re.search(pattern, text)
        return m.group(1) if m else text.strip().splitlines()[0][:40]
    except Exception:
        return ""


def workload_dims(workload: str) -> tuple[int, int, int]:
    """The matrix shape each workload presents to the accelerator.

    Derived from the frozen configs' largest layer, because that layer
    dominates the cycle count and is the one worth optimising.

      workload_a: the FC layer, 1024 -> 4, so M=1 N=4 K=256 (K in 4-byte words)
      workload_b: the largest FC, 32 -> 128, so M=1 N=128 K=8
    """
    if workload == "workload_a":
        return 1, 4, 256
    return 1, 128, 8


def build_verilator(array_h: int, array_w: int, wbuf: int, abuf: int,
                    precision: int, objdir: str) -> tuple[bool, str]:
    """Compile a Verilator simulator for one RTL configuration."""
    broot = space_free_root()  # see its docstring -- works around a
                               # Verilator Makefile bug when ROOT has a space
    cmd = [
        "verilator", "--cc", "--exe", "--build", "-j", "0",
        "-O3", "--x-assign", "fast", "--x-initial", "fast",
        "--top-module", "soc_top",
        "-Wno-fatal",
        "-I" + os.path.join(broot, "rtl"),
        "-GCLKS_PER_BIT=4",
        f"-GARRAY_H={array_h}", f"-GARRAY_W={array_w}",
        f"-GWBUF_DEPTH={wbuf}", f"-GABUF_DEPTH={abuf}",
        f"-GPRECISION={precision}",
        "--Mdir", objdir,
    ] + [os.path.join(broot, "rtl", f) for f in RTL_FILES] + [
        os.path.join(broot, "third_party", "picorv32", "picorv32.v"),
        os.path.join(broot, "sim", "verilator", "tb_soc.cpp"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=broot)
    if res.returncode != 0:
        return False, res.stderr[-500:]
    return True, ""


def golden_meta(workload: str, precision: int = 8) -> dict | None:
    """The Python reference's recorded output for this workload/precision.

    Each (workload, precision) pair has its OWN golden vector: quantizing to
    int4 changes the numbers, and workload B is a structurally different
    model, so checking one against another's golden file would be comparing
    against the wrong reference. Returns the whole metadata dict -- notably
    "task" ("classify" or "reconstruct") and either "predicted_class" or
    "expected_reconstruction_mae" -- because run_simulation needs to know
    WHICH check applies before it knows what to compare against.
    """
    import json
    cfg_path = os.path.join(ROOT, "train", "config", f"{workload}.yaml")
    if not os.path.exists(cfg_path):
        return None
    name = load_config(cfg_path).get("name")
    wl_suffix = "" if workload == "workload_a" else "_b"
    prec_suffix = "" if precision == 8 else f"_int{precision}"
    gdir = f"golden{wl_suffix}{prec_suffix}"
    meta = os.path.join(ROOT, "sim", gdir, f"{name}.json")
    if not os.path.exists(meta):
        return None
    with open(meta) as fh:
        return json.load(fh)


def run_simulation(array_h: int, array_w: int, wbuf: int, abuf: int,
                   precision: int, workload: str, timeout: int,
                   workdir: str, variant: str = "array",
                   golden: dict | None = None) -> dict:
    """Build and run one configuration, and CHECK IT GOT THE RIGHT ANSWER."""
    objdir = os.path.join(workdir, f"obj_{array_h}_{array_w}_{wbuf}_{precision}")

    ok, err = build_verilator(array_h, array_w, wbuf, abuf, precision, objdir)
    if not ok:
        return {"sim_ok": False, "error": "verilator build failed: " + err}

    # Firmware must match BOTH the hardware's precision AND the workload
    # being swept. Getting either wrong is a silent-wrong-answer bug, not a
    # crash: int4 hardware sign-extends the low nibble, so int8 weights would
    # mis-decode any value outside [-8,7]; and every workload's firmware
    # variant is named "baseline.hex"/"dot4.hex"/"array.hex", so loading
    # workload A's build while labelling the row "workload_b" would produce a
    # row that runs, completes, and reports someone else's numbers under the
    # wrong name. This naming scheme mirrors sw/Makefile's MODELS/BUILD
    # convention (see `make -C sw` -- MODELS=models_b, BUILD=build_b for
    # workload B) and Makefile's MODELS_DIR/GOLDEN_DIR for `make weights`.
    wl_suffix = "" if workload == "workload_a" else "_b"
    prec_suffix = "" if precision == 8 else f"_int{precision}"
    build_dir = f"build{wl_suffix}{prec_suffix}"
    models_dir = f"models{wl_suffix}{prec_suffix}"
    fw = os.path.join(ROOT, "sw", build_dir, f"{variant}.hex")
    if not os.path.exists(fw):
        return {"sim_ok": False,
                "error": f"{build_dir}/{variant}.hex not built -- run "
                         f"'make -C sw MODELS={models_dir} BUILD={build_dir}' "
                         f"(after 'make weights CONFIG=train/config/{workload}"
                         f".yaml ...' if {models_dir}/model_weights.h does "
                         f"not exist yet -- see the top-level Makefile)"}

    dst = os.path.join(ROOT, "sim", "build", "firmware.hex")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(fw, dst)

    try:
        res = subprocess.run(
            [os.path.join(objdir, "Vsoc_top"),
             # NOT +quiet: the performance counters are printed over the
             # simulated UART, so suppressing that output removes the very
             # data being collected.
             "+max_cycles=3000000000", "+progress=0"],
            capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"sim_ok": False, "error": "wall-clock timeout"}

    text = res.stdout + res.stderr
    fields: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("["):
            k, _, v = line.partition("=")
            if v.lstrip("-").isdigit():
                fields[k.strip()] = int(v.strip())

    if "cycles_total" not in fields:
        return {"sim_ok": False, "error": "no cycle counters in output"}

    cycles = fields["cycles_total"]
    accel_cycles = fields.get("accel_cycles", 0)
    accel_macs = fields.get("accel_macs", 0)
    peak = array_h * array_w * max(accel_cycles, 1)

    # The correctness check differs by TASK, not just by workload: a
    # classifier's firmware reports "class" (see sw/src/main.c's
    # MODEL_TASK_CLASSIFY branch); an autoencoder reports "reconstruction_mae"
    # instead, because an index into a reconstruction vector isn't a
    # meaningful answer to check. Comparing the wrong field would either
    # crash (KeyError-shaped None mismatch) or, worse, silently mark every
    # workload_b row "WRONG ANSWER" because "class" is never printed for a
    # reconstruction task -- which is exactly the kind of fast-and-wrong
    # result this check exists to catch, just aimed at itself instead of the
    # hardware. Both comparisons are bit-exact (deterministic integer
    # arithmetic on both sides), never a tolerance.
    task = (golden or {}).get("task", "classify")
    if task == "reconstruct":
        expect_val = (golden or {}).get("expected_reconstruction_mae")
        got_val = fields.get("reconstruction_mae")
    else:
        expect_val = (golden or {}).get("predicted_class")
        got_val = fields.get("class")
    correct = (expect_val is None) or (got_val == expect_val)

    return {
        "sim_ok": True,
        "cycles_total": cycles,
        "cycles_active": fields.get("cycles_mac", 0),
        "cycles_stall": fields.get("accel_stalls", 0),
        "macs_done": accel_macs,
        "accel_cycles": accel_cycles,
        "utilisation": round(accel_macs / peak, 6) if peak else 0.0,
        "accel_busy_fraction": round(accel_cycles / max(cycles, 1), 6),
        "stall_fraction": round(fields.get("accel_stalls", 0)
                                / max(accel_cycles, 1), 6),
        "predicted_class": got_val,
        "expected_class": expect_val,
        # A configuration that is fast and WRONG must never be reported as
        # fast. This flag is what stops that happening at scale.
        "result_correct": correct,
        "accel_timeouts": fields.get("accel_timeouts", 0),
    }


def run_synthesis(array_h: int, array_w: int, wbuf: int, abuf: int,
                  precision: int, part: str, target_mhz: float,
                  seed: int, workdir: str) -> dict:
    """Run Vivado in non-project mode and parse the utilisation/timing reports."""
    vivado = shutil.which("vivado")
    if vivado is None:
        return {"synth_ok": False, "error": "vivado not found"}

    outdir = os.path.join(workdir, f"synth_{array_w}_{wbuf}_{precision}_{seed}")
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        vivado, "-mode", "batch", "-nojournal", "-nolog",
        "-source", os.path.join(HERE, "vivado", "build.tcl"),
        "-tclargs",
        f"-part={part}", f"-array_h={array_h}", f"-array_w={array_w}",
        f"-wbuf={wbuf}", f"-abuf={abuf}", f"-precision={precision}",
        f"-period={1000.0/target_mhz:.3f}", f"-seed={seed}",
        f"-outdir={outdir}", f"-root={ROOT}",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"synth_ok": False, "error": "vivado timeout"}

    if res.returncode != 0:
        return {"synth_ok": False, "error": res.stdout[-400:]}

    summary = os.path.join(outdir, "summary.txt")
    if not os.path.exists(summary):
        return {"synth_ok": False, "error": "no summary.txt produced"}

    vals: dict = {}
    with open(summary) as fh:
        for line in fh:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                vals[k.strip()] = v.strip()

    try:
        wns = float(vals.get("WNS", "nan"))
        period = 1000.0 / target_mhz
        return {
            "synth_ok": True,
            "luts": int(float(vals.get("LUT", 0))),
            "ffs": int(float(vals.get("FF", 0))),
            "dsps": int(float(vals.get("DSP", 0))),
            "brams": float(vals.get("BRAM", 0)),
            "wns_ns": wns,
            # Fmax from slack: achieved period = target - slack.
            "fmax_mhz": round(1000.0 / (period - wns), 2) if period > wns else 0.0,
            "timing_met": wns >= 0,
        }
    except (ValueError, ZeroDivisionError) as exc:
        return {"synth_ok": False, "error": f"could not parse reports: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "sweep_config.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="simulation only; skip Vivado synthesis")
    ap.add_argument("--quick", action="store_true",
                    help="a small subset, for checking the pipeline works")
    ap.add_argument("--out", default=None)
    ap.add_argument("--filter", action="append", default=[],
                    help="restrict a factor, e.g. --filter array_w=8")
    args = ap.parse_args()

    cfg = load_config(args.config)
    factors = cfg["factors"]

    array_ws = factors["array_w"]
    wbufs = factors["wbuf_depth"]
    precisions = factors["precision"]
    workloads = factors["workload"]

    if args.quick:
        # Depth 0 is excluded from the sweep -- it cannot compute a correct
        # result (docs/DECISIONS.md D016). 64 vs 256 straddles conv2's 72-word
        # working set, which is where the interesting effect lives.
        array_ws, wbufs, precisions = [1, 4], [64, 256], [8]
        workloads = workloads[:1]
        print("QUICK MODE: a subset only, for pipeline checking.")

    for f in args.filter:
        key, _, val = f.partition("=")
        if key == "array_w":     array_ws = [int(val)]
        elif key == "wbuf_depth": wbufs = [int(val)]
        elif key == "precision":  precisions = [int(val)]
        elif key == "workload":   workloads = [val]
        else:
            ap.error(f"unknown filter key '{key}'")

    replicates = cfg.get("replicates", 1)
    # Simulation is deterministic, so replicates measure nothing here.
    if replicates > 1:
        print(f"NOTE: replicates collapsed {replicates} -> 1 for simulation.")
        print("      RTL simulation is deterministic; repeating it would")
        print("      manufacture an appearance of precision that is not real.")
        print("      Replicates exist for Vivado's non-deterministic placer.")
        replicates = 1

    out_csv = args.out or os.path.join(ROOT, cfg["output"]["csv"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if shutil.which("iverilog") is None:
        print("ERROR: iverilog not found -- nothing can be simulated.\n"
              "  macOS: brew install icarus-verilog\n"
              "  Ubuntu: sudo apt-get install iverilog", file=sys.stderr)
        return 127

    iv_ver = tool_version(["iverilog", "-V"], r"Icarus Verilog version ([\d.]+)")
    viv_ver = "" if args.dry_run else tool_version(["vivado", "-version"],
                                                   r"Vivado v([\d.]+)")
    sha = git_sha()

    # Frozen config hashes, so each row can be traced to an exact model.
    wl_hashes, wl_synth = {}, {}
    for wl in workloads:
        p = os.path.join(ROOT, "train", "config", f"{wl}.yaml")
        if os.path.exists(p):
            wc = load_config(p)
            wl_hashes[wl] = config_hash(wc)
            golden = os.path.join(ROOT, "sim", "golden", f"{wc['name']}.json")
            if os.path.exists(golden):
                import json
                with open(golden) as fh:
                    wl_synth[wl] = bool(json.load(fh).get("synthetic", False))

    cells = list(itertools.product(array_ws, wbufs, precisions, workloads))
    total = len(cells) * replicates
    print(f"Sweeping {len(cells)} configurations x {replicates} replicate(s) "
          f"= {total} runs")
    print(f"  synthesis: {'DISABLED (--dry-run)' if args.dry_run else 'enabled'}")
    print(f"  output:    {out_csv}")
    print()

    wl_golden = {(wl, pr): golden_meta(wl, pr)
                 for wl in workloads for pr in precisions}
    print(f"  golden references: "
          f"{ {k: (v or {}).get('predicted_class', (v or {}).get('expected_reconstruction_mae')) for k, v in wl_golden.items()} }")
    print(f"  firmware variant: {VARIANT}")
    print()

    rows, n_done, n_fail, n_wrong = [], 0, 0, 0
    with tempfile.TemporaryDirectory(prefix="sweep_") as workdir:
        for (aw, wb, prec, wl) in cells:
            ah = aw if cfg.get("array_h_follows_w", True) else aw
            for rep in range(replicates):
                n_done += 1
                sim = run_simulation(ah, aw, wb, wb, prec, wl,
                                     cfg["simulation"]["timeout_seconds"],
                                     workdir, VARIANT,
                                     wl_golden.get((wl, prec)))
                syn = ({"synth_ok": False} if args.dry_run else
                       run_synthesis(ah, aw, wb, wb, prec,
                                     cfg["synthesis"]["part"],
                                     cfg["synthesis"]["target_clock_mhz"],
                                     rep, workdir))

                if not sim.get("sim_ok"):
                    n_fail += 1
                    print(f"  [{n_done}/{total}] FAIL {aw}x{ah} wbuf={wb} "
                          f"p={prec} {wl}: {sim.get('error','?')}")
                else:
                    ok = sim.get("result_correct", True)
                    if not ok:
                        n_wrong += 1
                    print(f"  [{n_done}/{total}] {aw}x{ah} wbuf={wb:5d} "
                          f"p={prec} {wl:10s} "
                          f"cycles={sim['cycles_total']:>11,} "
                          f"busy={100*sim['accel_busy_fraction']:5.2f}% "
                          f"{'OK' if ok else 'WRONG ANSWER'}")

                rows.append({
                    "array_w": aw, "array_h": ah,
                    "wbuf_depth": wb, "abuf_depth": wb,
                    "precision": prec, "workload": wl, "replicate": rep,
                    "cycles_total": sim.get("cycles_total", ""),
                    "cycles_active": sim.get("cycles_active", ""),
                    "cycles_stall": sim.get("cycles_stall", ""),
                    "macs_done": sim.get("macs_done", ""),
                    "accel_cycles": sim.get("accel_cycles", ""),
                    "utilisation": sim.get("utilisation", ""),
                    "accel_busy_fraction": sim.get("accel_busy_fraction", ""),
                    "stall_fraction": sim.get("stall_fraction", ""),
                    "sim_ok": sim.get("sim_ok", False),
                    "predicted_class": sim.get("predicted_class", ""),
                    "expected_class": sim.get("expected_class", ""),
                    "result_correct": sim.get("result_correct", ""),
                    "accel_timeouts": sim.get("accel_timeouts", ""),
                    "luts": syn.get("luts", ""), "ffs": syn.get("ffs", ""),
                    "dsps": syn.get("dsps", ""), "brams": syn.get("brams", ""),
                    "fmax_mhz": syn.get("fmax_mhz", ""),
                    "wns_ns": syn.get("wns_ns", ""),
                    "timing_met": syn.get("timing_met", ""),
                    "synth_ok": syn.get("synth_ok", False),
                    # Accuracy is filled in by the quantization pipeline, not
                    # measured here. Left blank rather than zero so it can
                    # never be mistaken for "0% accurate".
                    "accuracy": "",
                    "accuracy_is_synthetic": wl_synth.get(wl, ""),
                    # Energy needs the PPK2, which does not exist yet.
                    "energy_per_inference_uj": "TBD_MEASURED",
                    "energy_source": "TBD_MEASURED",
                    "git_sha": sha,
                    "timestamp_utc": datetime.datetime.now(
                        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "iverilog_version": iv_ver,
                    "vivado_version": viv_ver,
                    "config_sha256": wl_hashes.get(wl, ""),
                    "seed": rep,
                })

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"Wrote {out_csv}: {len(rows)} rows, {n_fail} failed, "
          f"{n_wrong} produced a WRONG ANSWER")
    if n_wrong:
        print("  Rows with result_correct=False are configurations that ran")
        print("  but computed the wrong classification. Their cycle counts are")
        print("  NOT valid performance results and must be excluded.")
    if n_fail:
        print("  Rows with sim_ok=False must be EXCLUDED from analysis, not")
        print("  treated as zeros. See sweep/results/SCHEMA.md.")
    print()
    print("  energy_per_inference_uj is TBD_MEASURED for every row: the Power")
    print("  Profiler Kit II has not been purchased, and this project does not")
    print("  substitute estimates for measurements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
