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
#   ./sim/run_verilator.sh lint          # lint every module (recommended)
#   ./sim/run_verilator.sh lint soc_top  # lint one module
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

case "$MODE" in
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
        echo "Unknown mode: $MODE (expected 'lint')"; exit 2 ;;
esac
