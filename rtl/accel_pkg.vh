//===========================================================================
// accel_pkg.vh -- Shared parameters, opcodes and helper macros
//===========================================================================
//
// WHAT THIS FILE IS, for someone who has never written Verilog:
//
//   Verilog-2001 (the dialect we use, because every tool supports it --
//   Icarus, Verilator and Vivado) has no "package" or "import" like Python.
//   The only way to share constants between files is textual inclusion:
//   `include "accel_pkg.vh" literally pastes this file in at that point.
//
//   That means: define ONLY constants and macros here. No modules, no logic.
//   If you put a module in here and include it twice, you get a duplicate
//   module error that is confusing to debug.
//
//   The include guard below (`ifndef / `define / `endif) makes double
//   inclusion harmless -- exactly like #pragma once in C.
//
// WHY THESE ARE PARAMETERS AND NOT NUMBERS:
//
//   The entire research question is "how do these knobs trade off against
//   each other?". If any of ARRAY_W / WBUF_DEPTH / PRECISION were hardcoded
//   anywhere, the sweep in sweep/run_sweep.py would silently build the same
//   design N times and report a flat line. Every knob below is overridable
//   from the command line at build time:
//
//     iverilog -PARRAY_W=4 ...          (Icarus, per-instance)
//     $ verilator -GARRAY_W=4 ...         (Verilator)
//     set_property generic {ARRAY_W=4}  (Vivado)
//
//   The defaults here are only defaults. Never read a default and assume the
//   build used it -- check the run's metadata in the sweep CSV.
//===========================================================================

`ifndef ACCEL_PKG_VH
`define ACCEL_PKG_VH

//---------------------------------------------------------------------------
// Custom instruction encoding (RISC-V "custom-0" space)
//---------------------------------------------------------------------------
// The RISC-V spec reserves four opcodes for non-standard extensions so that
// vendors can add instructions without ever colliding with a future official
// one. custom-0 is 7'b0001011 (0x0B). We use the R-type layout:
//
//   31      25 24   20 19   15 14  12 11   7 6      0
//  +----------+-------+-------+------+------+--------+
//  |  funct7  |  rs2  |  rs1  |funct3|  rd  | opcode |
//  +----------+-------+-------+------+------+--------+
//
// PicoRV32 hands us the whole 32-bit instruction word plus the already-read
// rs1/rs2 register VALUES, so we only ever decode funct7/funct3/opcode.
//---------------------------------------------------------------------------
`define ACCEL_OPCODE_CUSTOM0   7'b0001011

// funct3 is 0 for all three of our instructions; funct7 selects the operation.
`define ACCEL_FUNCT3_DOT       3'b000

`define ACCEL_FUNCT7_DOT4      7'b0000000  // rd = dot4(rs1, rs2)
`define ACCEL_FUNCT7_DOT4A     7'b0000001  // acc += dot4(rs1, rs2); rd unwritten
`define ACCEL_FUNCT7_ACCRD     7'b0000010  // rd = acc; acc = 0

//---------------------------------------------------------------------------
// Datapath widths
//---------------------------------------------------------------------------
// A "MAC" multiplies two 8-bit signed values (range -128..127) and adds the
// result into a wide accumulator. Why 32 bits of accumulator for 8-bit
// inputs? Because products are up to 16 bits and we add thousands of them:
//   worst case |(-128) * (-128)| = 16384, times 4096 accumulations = 2^26.
// 32 bits gives generous headroom so we never have to reason about overflow
// in the inner loop. Overflow here would be a silent wrong-answer bug.
`define ACCEL_ACC_W            32
`define ACCEL_OPERAND_W        8    // container width; see PRECISION below
`define ACCEL_PROD_W           16   // 8x8 signed product needs 16 bits

// Number of int8 lanes packed into one 32-bit RISC-V register.
// 32 / 8 = 4, which is why the instruction is called DOT4.
`define ACCEL_DOT_LANES        4

//---------------------------------------------------------------------------
// Memory map for the accelerator's control registers
//---------------------------------------------------------------------------
// The MAC array is memory-mapped rather than sitting on PCPI. Reason (this
// is a real design decision, see docs/DECISIONS.md): PCPI STALLS THE CORE
// while the coprocessor works. That is fine for a 1-cycle dot product, but
// if the array runs for 500 cycles the core sits idle for 500 cycles and we
// have gained nothing. Memory-mapped means the core writes a "go" bit and is
// then free -- it can poll, sleep, or prepare the next tile.
//
// These offsets are relative to ACCEL_BASE_ADDR. Keep them in sync with
// sw/include/accel.h -- a mismatch here is a classic silent bug where the
// software writes the length register into the start register.
`define ACCEL_BASE_ADDR        32'h4000_0000

`define ACCEL_REG_CTRL         8'h00  // W: bit0=start(pulse) bit1=soft_reset
`define ACCEL_REG_STATUS       8'h04  // R: bit0=busy bit1=done bit2=result_valid
`define ACCEL_REG_M            8'h08  // RW: output rows    (M)
`define ACCEL_REG_N            8'h0C  // RW: output columns (N)
`define ACCEL_REG_K            8'h10  // RW: reduction depth(K)
`define ACCEL_REG_WBUF         8'h40  // W : weight buffer write port
`define ACCEL_REG_ABUF         8'h44  // W : activation buffer write port
`define ACCEL_REG_RESULT       8'h48  // R : pop one int32 result from FIFO
`define ACCEL_REG_CYCLES       8'h50  // R : perf counter -- total cycles
`define ACCEL_REG_STALLS       8'h54  // R : perf counter -- data-starved cycles
`define ACCEL_REG_MACS         8'h58  // R : perf counter -- useful MACs issued

//---------------------------------------------------------------------------
// Requantization constants
//---------------------------------------------------------------------------
// See docs/ARCHITECTURE.md "Requantization" for the full derivation. Short
// version: to scale an int32 accumulator back down to int8 we multiply by a
// fixed-point multiplier M0 and then do a rounding right shift. M0 is
// normalized to sit in [2^30, 2^31), which is why the base shift is 31.
//
// Doing this in floating point instead would make bit-exact agreement
// between PyTorch, C and Verilog effectively impossible, because the three
// would round differently in the last bit.
`define ACCEL_M0_SHIFT         31

//---------------------------------------------------------------------------
// Saturation limits, as a function of PRECISION
//---------------------------------------------------------------------------
// PRECISION is a module parameter (8 or 4), so these have to be expressions
// evaluated per-module, not `defines. They are written here as macros that
// take the parameter name, so every module clamps identically.
//
//   PRECISION=8 -> [-128, 127]
//   PRECISION=4 -> [   -8,   7]
`define ACCEL_QMIN(P)          (-(1 <<< ((P)-1)))
`define ACCEL_QMAX(P)          ((1 <<< ((P)-1)) - 1)

`endif // ACCEL_PKG_VH
