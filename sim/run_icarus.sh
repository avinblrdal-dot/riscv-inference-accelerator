#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_icarus.sh -- compile and run one testbench under Icarus Verilog
#
# Usage:
#   ./sim/run_icarus.sh                    # run every testbench
#   ./sim/run_icarus.sh tb_dot4            # run one
#   ./sim/run_icarus.sh tb_soc dot4        # run tb_soc with the dot4 firmware
#   VCD=1 ./sim/run_icarus.sh tb_dot4      # also dump a waveform
#
# Why a script and not just a long iverilog command: the file list and the
# include path have to be identical for every testbench, and getting them
# subtly different between runs is a great way to spend an afternoon
# debugging a "bug" that is actually a stale build.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="$ROOT/sim/build"
mkdir -p "$BUILD"

if ! command -v iverilog >/dev/null 2>&1; then
    echo "ERROR: iverilog not found."
    echo "  macOS:  brew install icarus-verilog"
    echo "  Ubuntu: sudo apt-get install iverilog"
    echo "  See docs/GETTING_STARTED.md for full setup."
    exit 127
fi

RTL="$ROOT/rtl"
PICORV32="$ROOT/third_party/picorv32/picorv32.v"

if [ ! -f "$PICORV32" ]; then
    echo "ERROR: PicoRV32 submodule is missing."
    echo "  Fix with:  git submodule update --init --recursive"
    exit 1
fi

# Every RTL source. Order does not matter to Icarus, but keeping the list in
# one place does.
RTL_SRCS=(
    "$RTL/dot4_pcpi.v" "$RTL/mac_unit.v" "$RTL/mac_array.v"
    "$RTL/weight_buffer.v" "$RTL/activation_buffer.v" "$RTL/accel_ctrl.v"
    "$RTL/accel_top.v" "$RTL/perf_counter.v" "$RTL/requantize.v"
    "$RTL/uart_tx.v" "$RTL/soc_top.v"
)

VCD_FLAG=""
[ "${VCD:-0}" = "1" ] && VCD_FLAG="+vcd"

run_one () {
    local tb="$1"
    local fw="${2:-boot}"
    local tb_file="$ROOT/sim/${tb}.v"

    if [ ! -f "$tb_file" ]; then
        echo "ERROR: no such testbench: $tb_file"; return 1
    fi

    echo ""
    echo "======================================================================"
    echo " $tb"
    echo "======================================================================"

    local srcs=("$tb_file")
    # Only the full-system testbench needs the CPU and the whole RTL set;
    # unit testbenches compile just what they exercise, which keeps them fast
    # and means a break in one module cannot mask a break in another.
    case "$tb" in
        tb_soc)
            python3 "$ROOT/sim/gen_smoke_hex.py" --test "$fw" -o "$BUILD/${fw}.hex"
            # tb_soc always loads sim/build/firmware.hex -- see the comment
            # in tb_soc.v for why the name is fixed rather than parameterised.
            cp "$BUILD/${fw}.hex" "$BUILD/firmware.hex"
            srcs+=("${RTL_SRCS[@]}" "$PICORV32")
            ;;
        tb_dot4)          srcs+=("$RTL/dot4_pcpi.v") ;;
        tb_mac_array)     srcs+=("$RTL/mac_array.v" "$RTL/mac_unit.v") ;;
        tb_weight_buffer) srcs+=("$RTL/weight_buffer.v") ;;
        *)                srcs+=("${RTL_SRCS[@]}" "$PICORV32") ;;
    esac

    # Warnings from third_party/ are not ours to fix, and the "not enough
    # words" note just means the program is smaller than the RAM, which is
    # normal. Filtering them keeps real warnings visible.
    iverilog -g2005 -Wall -I"$RTL" -o "$BUILD/${tb}.vvp" "${srcs[@]}" 2>&1 \
        | grep -v "third_party/" | grep -v "cpuregs" || true

    local out
    out="$(cd "$ROOT" && vvp "$BUILD/${tb}.vvp" $VCD_FLAG 2>&1 \
           | grep -v "Not enough words in the file")"
    echo "$out"

    # A testbench passes only if it SAYS it passed. Relying on the exit code
    # alone is not enough: a Verilog $finish always exits 0, even after the
    # testbench has printed a hundred failures.
    if echo "$out" | grep -q "TEST FAILED"; then
        echo ">>> $tb FAILED"; return 1
    fi
    if echo "$out" | grep -q "TEST PASSED"; then
        echo ">>> $tb passed"; return 0
    fi
    # tb_soc has no explicit PASSED line; it is judged on its printed output.
    if [ "$tb" = "tb_soc" ]; then
        case "$fw" in
            boot) echo "$out" | grep -q "RVACCEL-BOOT-OK" \
                    && { echo ">>> tb_soc/boot passed"; return 0; } ;;
            dot4) echo "$out" | grep -q "123" \
                    && { echo ">>> tb_soc/dot4 passed"; return 0; } ;;
        esac
        echo ">>> tb_soc/$fw FAILED (expected output not seen)"; return 1
    fi
    echo ">>> $tb produced no verdict"; return 1
}

FAILED=0
if [ $# -ge 1 ]; then
    run_one "$@" || FAILED=1
else
    for tb in tb_dot4 tb_mac_array tb_weight_buffer; do
        run_one "$tb" || FAILED=1
    done
    run_one tb_soc boot || FAILED=1
    run_one tb_soc dot4 || FAILED=1
fi

echo ""
if [ "$FAILED" = "0" ]; then echo "ALL SIMULATION TESTS PASSED"; else echo "SOME SIMULATION TESTS FAILED"; fi
exit $FAILED
