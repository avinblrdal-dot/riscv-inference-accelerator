`timescale 1ns / 1ps
//===========================================================================
// requantize.v -- int32 accumulator -> int8 output, integer-only
//===========================================================================
//
// THIS IS THE MOST IMPORTANT MODULE FOR BIT-EXACTNESS.
//
//   The identical arithmetic is implemented three times in this project:
//     Python  train/quant_ref.py   requantize()
//     C       sw/src/quant.c       requantize()
//     Verilog THIS FILE
//   All three must produce THE SAME BITS on every input. train/verify_parity.py
//   proves it. If they ever diverge, a hardware bug becomes indistinguishable
//   from numerical drift and the project loses its ability to debug itself.
//
// WHY NOT JUST USE FLOATING POINT?
//
//   Conceptually we want:  out = round(acc * (s_in * s_w / s_out))
//   where the s's are the quantization scales. That scale factor is a real
//   number like 0.0023419. Doing it in float would mean: PyTorch uses
//   float64, C on the host uses float64, the RISC-V core has NO floating
//   point unit at all, and the FPGA would need a float multiplier costing
//   more area than the entire MAC array. Worse, the three would round
//   differently in the last bit, so "bit-exact" would be unachievable.
//
//   So we do what TFLite Micro and CMSIS-NN do: convert the real multiplier
//   ONCE, offline, into a pair of integers (M0, n), and then use only
//   integer multiply and shift at runtime.
//
// THE DECOMPOSITION (done offline in train/quantize.py):
//
//   real_multiplier = s_in * s_w / s_out,  which is in (0, 1)
//   Write it as     m * 2^e   with m in [0.5, 1) and e <= 0   (frexp)
//   Then            M0 = round(m * 2^31)   so M0 is in [2^30, 2^31)
//                   n  = -e                so n >= 0
//   And             acc * real_multiplier  ==  (acc * M0) >> (31 + n)
//
//   Edge case: if m rounds up to exactly 1.0, M0 would be 2^31 which does
//   not fit in a signed 32-bit integer. The fix, applied in Python, is to
//   halve M0 and decrement n. See quantize.py.
//
// THE ROUNDING RULE -- READ THIS BEFORE CHANGING ANYTHING
//
//   We use ROUND HALF AWAY FROM ZERO, applied symmetrically:
//
//     prod  = acc * M0                     (64-bit signed intermediate)
//     shift = 31 + n
//     half  = 1 << (shift - 1)
//     if prod >= 0:  out =  ( prod + half) >>> shift
//     else:          out = -((-prod + half) >>> shift)
//
//   Why the explicit negate-shift-negate for negatives instead of just
//   (prod + half) >>> shift? Because an arithmetic right shift rounds toward
//   NEGATIVE INFINITY, so adding half would round -2.5 to -2 but +2.5 to +3
//   -- asymmetric. Asymmetric rounding puts a small DC bias into every
//   layer's output, which accumulates across layers and shows up as a real
//   accuracy loss. The negate trick makes the operation symmetric about
//   zero: -2.5 -> -3 and +2.5 -> +3.
//
//   NOTE this differs slightly from gemmlowp's exact "nudge" formulation.
//   That is a deliberate choice (see docs/DECISIONS.md, "Rounding rule"):
//   gemmlowp's version is harder to reproduce identically in three languages,
//   and what matters scientifically is that OUR three implementations agree
//   with each other and that the rule is documented -- not that we match
//   Google's last bit. If you ever want to compare directly against a TFLite
//   reference number, this is the line you will need to revisit.
//
// PARAMETERS:
//   PRECISION  8 -> clamp to [-128,127];  4 -> clamp to [-8,7]
//
// PORTS: purely combinational. acc/mult/shift in, clamped value out.
//===========================================================================

`include "accel_pkg.vh"

module requantize #(
    parameter PRECISION = 8
) (
    input  wire signed [31:0] acc,        // accumulator from the MAC array
    input  wire signed [31:0] mult,       // M0, normalized to [2^30, 2^31)
    input  wire        [5:0]  shift,      // total right shift = 31 + n
    input  wire signed [31:0] zero_point, // added after scaling (0 for symmetric)
    output wire signed [31:0] q_out       // clamped result, sign-extended
);

    localparam signed [31:0] QMIN = -(1 <<< (PRECISION-1));
    localparam signed [31:0] QMAX =  (1 <<< (PRECISION-1)) - 1;

    // 64-bit intermediate. acc is 32 bits and M0 is up to 2^31, so the
    // product needs up to 63 bits. Using 32 here would overflow on almost
    // every input -- a silent, catastrophic, and very easy mistake.
    wire signed [63:0] prod = $signed(acc) * $signed(mult);

    // half = 1 << (shift-1). Built as a 64-bit shift so it is correct for
    // any legal shift value.
    wire signed [63:0] half = (64'sd1 <<< (shift - 6'd1));

    // The symmetric rounding shift described above.
    wire signed [63:0] prod_abs   = (prod >= 0) ? prod : -prod;
    wire signed [63:0] shifted_abs = (prod_abs + half) >>> shift;
    wire signed [63:0] shifted    = (prod >= 0) ? shifted_abs : -shifted_abs;

    // Add the zero point, then clamp (saturate). Saturation, not wraparound:
    // if a value overflows we want the nearest representable number, because
    // wrapping would turn a large positive activation into a large negative
    // one and produce a wildly wrong classification.
    wire signed [63:0] biased = shifted + $signed({{32{zero_point[31]}}, zero_point});

    assign q_out = (biased < QMIN) ? QMIN :
                   (biased > QMAX) ? QMAX :
                                     biased[31:0];

endmodule
