#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_verilator.sh -- lint and (optionally) simulate with Verilator
#
# Verilator is a different kind of tool from Icarus. Icarus INTERPRETS Verilog;
# Verilator COMPILES it into C++ and builds a fast binary. Two consequences:
#
#   * Verilator is 10-100x faster, which matters once the sweep runs 48
#     configurations x several workloads.
#   * Verilator is far stricter. It rejects things Icarus quietly accepts --
#     width mismatches, unused signals, combinational loops, latches you did
#     not mean to create. That strictness is genuinely useful: most of what
#     Verilator complains about is a real bug waiting to happen on hardware.
#
# So even if you never run a Verilator simulation, RUNNING THE LINT IS WORTH
# IT. Do it before every commit that touches RTL:
#
#     ./sim/run_verilator.sh lint
#
# Usage:
#   ./sim/run_verilator.sh lint            # lint every module (recommended)
#   ./sim/run_verilator.sh lint soc_top    # lint one module
#   ./sim/run_verilator.sh build           # compile the fast simulator
#   ./sim/run_verilator.sh sim             # build (if needed) and run
#   ./sim/run_verilator.sh sim baseline    # run a specific firmware variant
#
# WHY THE `sim` MODE MATTERS
#   One baseline inference is ~250 million cycles, because rv32i has no
#   hardware multiply and every `a * b` -- including the address arithmetic
#   in the convolution loops -- becomes a libgcc __mulsi3 call. Under Icarus
#   that is about an hour. The design-space sweep is 64 configurations, so at
#   Icarus speed the core experiment of this project would take months.
#   Verilator compiles the design to C++ and makes it feasible.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RTL="$ROOT/rtl"

if ! command -v verilator >/dev/null 2>&1; then
    echo "ERROR: verilator not found."
    echo "  macOS:  brew install verilator"
    echo "  Ubuntu: sudo apt-get install verilator"
    echo ""
    echo "Verilator is OPTIONAL -- ./sim/run_icarus.sh covers all the"
    echo "functional tests. Verilator adds stricter linting and much faster"
    echo "simulation for the sweep. See docs/GETTING_STARTED.md."
    exit 127
fi

MODE="${1:-lint}"
TOP="${2:-}"

RTL_SRCS=(
    "$RTL/dot4_pcpi.v" "$RTL/mac_unit.v" "$RTL/mac_array.v"
    "$RTL/weight_buffer.v" "$RTL/activation_buffer.v" "$RTL/accel_ctrl.v"
    "$RTL/accel_top.v" "$RTL/perf_counter.v" "$RTL/requantize.v"
    "$RTL/uart_tx.v" "$RTL/soc_top.v"
)

# Warnings we deliberately silence, with reasons:
#   DECLFILENAME  our filenames match modules except for helper modules
#   UNUSED/UNDRIVEN  intentional on tie-off ports in the DEPTH=0 control case
#   WIDTHTRUNC/WIDTHEXPAND  reviewed case by case; parameter arithmetic in
#                 generate blocks trips these constantly without being wrong
#   PINCONNECTEMPTY  unconnected outputs we genuinely do not use
#   GENUNNAMED    unnamed generate blocks inside third_party/picorv32. Not
#                 our code, and a naming-style rule from IEEE 1800-2023
#                 rather than a defect.
#   BLKSEQ        blocking '=' in picorv32's sequential blocks. Deliberate in
#                 that codebase and not ours to change. OUR RTL must never
#                 trip this -- if it ever does, remove this waiver and fix it,
#                 because blocking assignments in clocked logic are a real bug
#                 (see docs/VERILOG_PRIMER.md section 4).
WAIVERS="-Wno-DECLFILENAME -Wno-UNUSED -Wno-UNDRIVEN -Wno-PINCONNECTEMPTY -Wno-GENUNNAMED -Wno-BLKSEQ"

lint_one () {
    local top="$1"
    echo "--- verilator lint: $top ---"
    verilator --lint-only -Wall $WAIVERS -I"$RTL" --top-module "$top" \
        "${RTL_SRCS[@]}" "$ROOT/third_party/picorv32/picorv32.v" \
        || return 1
}

# Parameters the simulator is built with. CLKS_PER_BIT is deliberately small
# so the UART costs microseconds of simulated time rather than milliseconds;
# the C++ receiver is told the same value so the two stay in step.
CLKS_PER_BIT="${CLKS_PER_BIT:-4}"
ARRAY_H="${ARRAY_H:-4}"
ARRAY_W="${ARRAY_W:-4}"
WBUF_DEPTH="${WBUF_DEPTH:-256}"
ABUF_DEPTH="${ABUF_DEPTH:-256}"
PRECISION="${PRECISION:-8}"

OBJDIR="$ROOT/sim/build/obj_dir"

build_sim () {
    echo "--- verilator build: ${ARRAY_H}x${ARRAY_W} wbuf=${WBUF_DEPTH} p=${PRECISION} ---"
    mkdir -p "$ROOT/sim/build"

    # -O3 and --x-assign fast trade simulation fidelity on undefined values
    # for speed. That is safe HERE because every register in this design is
    # reset (verified in docs/REVIEW.md section 1); on a design with unreset
    # state it would mask real bugs.
    verilator --cc --exe --build -j 0 \
        -O3 --x-assign fast --x-initial fast \
        --top-module soc_top \
        -Wall $WAIVERS \
        -Wno-fatal \
        -I"$RTL" \
        -GCLKS_PER_BIT=$CLKS_PER_BIT \
        -GARRAY_H=$ARRAY_H \
        -GARRAY_W=$ARRAY_W \
        -GWBUF_DEPTH=$WBUF_DEPTH \
        -GABUF_DEPTH=$ABUF_DEPTH \
        -GPRECISION=$PRECISION \
        --Mdir "$OBJDIR" \
        "${RTL_SRCS[@]}" "$ROOT/third_party/picorv32/picorv32.v" \
        "$ROOT/sim/verilator/tb_soc.cpp"
    echo "  built $OBJDIR/Vsoc_top"
}

run_sim () {
    local fw="${1:-}"
    mkdir -p "$ROOT/sim/build"

    # Pick the firmware. Prefer a real cross-compiled variant; fall back to
    # the toolchain-free smoke program so this works with no RISC-V GCC.
    local src=""
    if [ -n "$fw" ] && [ -f "$ROOT/sw/build/${fw}.hex" ]; then
        src="$ROOT/sw/build/${fw}.hex"
    elif [ -n "$fw" ]; then
        python3 "$ROOT/sim/gen_smoke_hex.py" --test "$fw" \
            -o "$ROOT/sim/build/${fw}.hex" >/dev/null
        src="$ROOT/sim/build/${fw}.hex"
    elif [ -f "$ROOT/sw/build/baseline.hex" ]; then
        src="$ROOT/sw/build/baseline.hex"
    else
        python3 "$ROOT/sim/gen_smoke_hex.py" --test boot \
            -o "$ROOT/sim/build/boot.hex" >/dev/null
        src="$ROOT/sim/build/boot.hex"
    fi

    # soc_top always $readmemh's this fixed filename -- see tb_soc.v for why
    # the name is fixed rather than parameterised.
    cp "$src" "$ROOT/sim/build/firmware.hex"
    echo "--- running $(basename "$src") ---"

    shift 2>/dev/null || true
    ( cd "$ROOT" && "$OBJDIR/Vsoc_top" "$@" )
}

case "$MODE" in
    build)
        build_sim
        ;;
    sim)
        [ -x "$OBJDIR/Vsoc_top" ] || build_sim
        run_sim "${2:-}" "${@:3}"
        ;;
    lint)
        if [ -n "$TOP" ]; then
            lint_one "$TOP"
        else
            FAILED=0
            for m in dot4_pcpi mac_unit mac_array weight_buffer activation_buffer \
                     accel_ctrl accel_top perf_counter requantize uart_tx soc_top; do
                lint_one "$m" || FAILED=1
            done
            [ "$FAILED" = "0" ] && echo "VERILATOR LINT CLEAN" || { echo "VERILATOR LINT FAILED"; exit 1; }
        fi
        ;;
    *)
        echo "Unknown mode: $MODE (expected lint, build or sim)"; exit 2 ;;
esac
