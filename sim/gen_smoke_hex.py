#!/usr/bin/env python3
"""Generate hand-assembled RV32I test firmware -- NO RISC-V TOOLCHAIN NEEDED.

Why this exists
---------------
The full software stack in ``sw/`` is written in C and needs a RISC-V
cross-compiler. Installing one is the single biggest setup hurdle for a new
team member, and until it is done nothing can be simulated at all -- which
makes it impossible to tell "my toolchain is broken" apart from "the RTL is
broken".

This script removes that dependency for the smoke tests. It contains a tiny
RV32I assembler (about 40 lines) and emits the same ``.hex`` format that
``$readmemh`` loads in ``rtl/soc_top.v``. So you can clone the repo, install
only Icarus Verilog, and immediately prove that:

  * the CPU comes out of reset and executes instructions
  * the memory bus and address decoder work
  * the UART transmits real characters
  * the DOT4 / DOT4A / ACCRD custom instructions work on the real core

If these pass but your C build fails, the problem is your toolchain. If these
fail, the problem is the RTL. That separation is worth the 40 lines.

Usage:
    python3 sim/gen_smoke_hex.py --test boot  -o sim/build/boot.hex
    python3 sim/gen_smoke_hex.py --test dot4  -o sim/build/dot4.hex
"""

from __future__ import annotations

import argparse
import os
from typing import List

# --- RV32I opcode constants (see the RISC-V unprivileged spec, Chapter 2) ---
OP_IMM = 0x13
LUI = 0x37
STORE = 0x23
LOAD = 0x03
BRANCH = 0x63
CUSTOM0 = 0x0B  # our custom instruction space

# Registers used by the helpers below. Kept deliberately separate from the
# operand registers so that printing a character cannot clobber a value under
# test -- a bug this script's author hit in exactly that way.
REG_UART_BASE = 1
REG_EXIT_BASE = 3
REG_SCRATCH_STATUS = 20
REG_SCRATCH_CHAR = 21


def enc_i(imm: int, rs1: int, f3: int, rd: int, op: int) -> int:
    """Encode an I-type instruction (immediate arithmetic, loads)."""
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def enc_s(imm: int, rs2: int, rs1: int, f3: int, op: int) -> int:
    """Encode an S-type instruction (stores). The immediate is split in two."""
    return (
        (((imm >> 5) & 0x7F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | ((imm & 0x1F) << 7)
        | op
    )


def enc_b(imm: int, rs2: int, rs1: int, f3: int, op: int) -> int:
    """Encode a B-type instruction (branches).

    The branch immediate is scrambled across the word in a way that looks
    bizarre until you know why: it keeps each immediate bit in the same
    physical position as the equivalent S-type bit, so the hardware can share
    decoder wiring. Bit 0 is always zero (branches are 2-byte aligned) and is
    not stored.
    """
    return (
        (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (f3 << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 1) << 7)
        | op
    )


def enc_u(imm20: int, rd: int, op: int) -> int:
    """Encode a U-type instruction (LUI: load upper 20 bits)."""
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | op


def enc_r(f7: int, rs2: int, rs1: int, f3: int, rd: int, op: int) -> int:
    """Encode an R-type instruction. Our custom instructions use this form."""
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def load_imm32(rd: int, val: int) -> List[int]:
    """Load an arbitrary 32-bit constant into ``rd`` using LUI + ADDI.

    The subtlety: ADDI sign-extends its 12-bit immediate. If the low 12 bits
    have bit 11 set, ADDI will subtract, so LUI must be pre-compensated by
    adding one to the upper part. Getting this wrong gives constants that are
    off by exactly 0x1000, which is a maddening bug to chase.
    """
    val &= 0xFFFFFFFF
    lo = val & 0xFFF
    signed_lo = lo - 4096 if lo & 0x800 else lo
    upper = ((val - signed_lo) >> 12) & 0xFFFFF
    return [enc_u(upper, rd, LUI), enc_i(signed_lo, rd, 0, rd, OP_IMM)]


def send_char(ch: int) -> List[int]:
    """Emit the instruction sequence that prints one character over the UART.

    Polls the busy flag first. Without the poll, characters written while the
    transmitter is still shifting are silently dropped and the output looks
    randomly truncated.
    """
    return [
        enc_i(4, REG_UART_BASE, 2, REG_SCRATCH_STATUS, LOAD),   # lw status
        enc_b(-4, 0, REG_SCRATCH_STATUS, 1, BRANCH),            # bne -> spin
        enc_i(ch, 0, 0, REG_SCRATCH_CHAR, OP_IMM),              # addi char
        enc_s(0, REG_SCRATCH_CHAR, REG_UART_BASE, 0, STORE),    # sb -> UART
    ]


def send_str(s: str) -> List[int]:
    out: List[int] = []
    for c in s.encode():
        out += send_char(c)
    return out


def check_eq(rs1: int, rs2: int, ok_char: str) -> List[int]:
    """Print ok_char if rs1 == rs2, otherwise print 'F'.

    Layout, so the branch offsets are auditable:
        +0   beq rs1, rs2, +24   (skip 6 instructions -> the ok path)
        +4   .. +16  send('F')   (4 instructions)
        +20  beq x0, x0, +20     (skip the ok path)
        +24  .. +36  send(ok)    (4 instructions)
    """
    return (
        [enc_b(4 * 6, rs2, rs1, 0, BRANCH)]
        + send_char(ord("F"))
        + [enc_b(4 * 5, 0, 0, 0, BRANCH)]
        + send_char(ord(ok_char))
    )


def prologue() -> List[int]:
    """Set up base-address registers for the UART and the sim-exit port."""
    return [
        enc_u(0x10000, REG_UART_BASE, LUI),   # 0x1000_0000
        enc_u(0x30000, REG_EXIT_BASE, LUI),   # 0x3000_0000
    ]


def epilogue() -> List[int]:
    """Write to the simulation exit port, which makes the testbench $finish."""
    return [enc_i(0, 0, 0, 6, OP_IMM), enc_s(0, 6, REG_EXIT_BASE, 2, STORE)]


def build_boot() -> List[int]:
    """Prove the CPU, memory, address decoder and UART all work."""
    return prologue() + send_str("RVACCEL-BOOT-OK\n") + epilogue()


def build_dot4() -> List[int]:
    """Exercise the three custom instructions on the real core.

    Expected output is exactly ``123``. Any ``F`` marks a failed subtest:
      1 = DOT4 computes a signed 4-lane dot product
      2 = DOT4A accumulates across instructions (regression for the
          double-accumulate bug -- see docs/DECISIONS.md)
      3 = ACCRD cleared the accumulator as a side effect
    """
    prog = prologue()

    # a lanes = [4, 3, 2, 1]; b lanes = [1, -1, 1, -1]
    # expected dot product = 4*1 + 3*(-1) + 2*1 + 1*(-1) = 2
    prog += load_imm32(5, 0x01020304)
    prog += load_imm32(6, 0xFF01FF01)

    # Test 1: DOT4 x7, x5, x6  ->  2
    prog += [enc_r(0b0000000, 6, 5, 0, 7, CUSTOM0), enc_i(2, 0, 0, 8, OP_IMM)]
    prog += check_eq(7, 8, "1")

    # Test 2: two DOT4A then ACCRD  ->  4  (not 8: each instruction counts once)
    prog += [
        enc_r(0b0000001, 6, 5, 0, 0, CUSTOM0),
        enc_r(0b0000001, 6, 5, 0, 0, CUSTOM0),
        enc_r(0b0000010, 0, 0, 0, 9, CUSTOM0),
        enc_i(4, 0, 0, 10, OP_IMM),
    ]
    prog += check_eq(9, 10, "2")

    # Test 3: the accumulator must have been cleared by ACCRD  ->  0
    prog += [enc_r(0b0000010, 0, 0, 0, 11, CUSTOM0)]
    prog += check_eq(11, 0, "3")

    prog += send_char(ord("\n"))
    return prog + epilogue()


TESTS = {"boot": build_boot, "dot4": build_dot4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", choices=sorted(TESTS), default="boot",
                    help="which smoke program to generate")
    ap.add_argument("-o", "--output", required=True, help="output .hex path")
    args = ap.parse_args()

    words = TESTS[args.test]()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("\n".join(f"{w:08x}" for w in words) + "\n")

    print(f"[gen_smoke_hex] test={args.test} "
          f"instructions={len(words)} -> {args.output}")


if __name__ == "__main__":
    main()
