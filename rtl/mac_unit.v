`timescale 1ns / 1ps
//===========================================================================
// mac_unit.v -- One multiply-accumulate cell
//===========================================================================
//
// THE CONCEPT:
//
//   "MAC" = Multiply-ACcumulate: acc = acc + (a * b). It is the single
//   operation that neural network inference is made of. A convolution or a
//   fully-connected layer is nothing but millions of these, arranged in
//   nested loops.
//
//   This module is one such cell. mac_array.v tiles ARRAY_W x ARRAY_H of
//   them so that many MACs happen in the SAME clock cycle. That parallelism
//   is the whole point of the accelerator: a CPU does one MAC per several
//   instructions, this does ARRAY_W*ARRAY_H per cycle.
//
// WHY THE ACCUMULATOR IS SO MUCH WIDER THAN THE INPUTS:
//
//   Inputs are 8-bit signed (-128..127). A single product fits in 16 bits.
//   But we add up to K of them, where K is the reduction depth of the layer
//   (often 1000+). 32 bits holds ~2 billion; the worst realistic case is
//   about 2^26, so we have plenty of margin. Overflow here would silently
//   produce a wrong classification, which is far worse than a crash because
//   nothing would tell you it happened.
//
// OUTPUT-STATIONARY DATAFLOW:
//
//   "Stationary" refers to which operand stays put in the cell while the
//   others stream past. Here the ACCUMULATOR stays in the cell and the
//   weights and activations stream in. The cell owns one output element of
//   the result matrix and builds it up over K cycles.
//
//   The alternative (weight-stationary) keeps a weight resident and streams
//   activations. This module is written so that swapping to weight-
//   stationary means changing mac_array.v's wiring and adding a weight
//   register here -- not rewriting the arithmetic. See the WEIGHT_REG
//   parameter, which is the hook for that future variant.
//
// PARAMETERS:
//   ACC_W      accumulator width in bits (default 32)
//   PRECISION  8 or 4. At 4, only the low nibble is meaningful and is
//              sign-extended; synthesis then infers a smaller multiplier,
//              which is the area saving the sweep measures.
//   WEIGHT_REG 0 = weight arrives combinationally each cycle (output
//                  stationary, the default)
//              1 = weight is latched by load_w and held (hook for a future
//                  weight-stationary variant)
//
// PORTS:
//   clk        clock
//   resetn     active-low reset
//   clr        synchronous clear of the accumulator (start a new output)
//   en         enable: only when high does a MAC actually happen. This is
//              what lets the controller stall the array without losing state.
//   load_w     latch a new stationary weight (only used when WEIGHT_REG=1)
//   a_in       activation, signed
//   b_in       weight, signed
//   acc_out    the running accumulator, readable at any time
//   mac_valid  high on cycles where a real MAC was performed. The perf
//              counter uses this to distinguish useful work from stalls,
//              which is the direct evidence for research question RQ3.
//===========================================================================

`include "accel_pkg.vh"

module mac_unit #(
    parameter ACC_W      = `ACCEL_ACC_W,
    parameter PRECISION  = 8,
    parameter WEIGHT_REG = 0
) (
    input  wire                    clk,
    input  wire                    resetn,
    input  wire                    clr,
    input  wire                    en,
    input  wire                    load_w,
    input  wire signed [7:0]       a_in,
    input  wire signed [7:0]       b_in,
    output wire signed [ACC_W-1:0] acc_out,
    output reg                     mac_valid
);

    //-----------------------------------------------------------------------
    // Operand conditioning for reduced precision
    //-----------------------------------------------------------------------
    // At PRECISION=4 the caller packs a 4-bit value into the low nibble.
    // We sign-extend it so the arithmetic below is unchanged. Note this is
    // the same idiom used in dot4_pcpi.v -- keep the two consistent, because
    // the parity harness compares their results against each other.
    wire signed [7:0] a_eff;
    wire signed [7:0] b_eff;

    generate
        if (PRECISION == 4) begin : g_p4
            assign a_eff = {{4{a_in[3]}}, a_in[3:0]};
            assign b_eff = {{4{b_in[3]}}, b_in[3:0]};
        end else begin : g_p8
            assign a_eff = a_in;
            assign b_eff = b_in;
        end
    endgenerate

    //-----------------------------------------------------------------------
    // Optional stationary weight register
    //-----------------------------------------------------------------------
    reg signed [7:0] w_held;
    always @(posedge clk) begin
        if (!resetn)      w_held <= 8'sd0;
        else if (load_w)  w_held <= b_eff;
    end

    wire signed [7:0] b_use = (WEIGHT_REG == 1) ? w_held : b_eff;

    //-----------------------------------------------------------------------
    // The multiply
    //-----------------------------------------------------------------------
    // Both operands are declared signed, so this is a signed multiply and
    // the 16-bit product is correct for negative inputs. On the Artix-7 this
    // maps onto a DSP48E1 slice (a hardened multiplier block) rather than
    // being built out of general-purpose logic -- which is why DSP count is
    // one of the area metrics we report.
    wire signed [`ACCEL_PROD_W-1:0] prod = a_eff * b_use;

    //-----------------------------------------------------------------------
    // The accumulate
    //-----------------------------------------------------------------------
    reg signed [ACC_W-1:0] acc;
    assign acc_out = acc;

    // Sign-extend the 16-bit product up to the accumulator width before
    // adding. Verilog would do this automatically here because both sides
    // are signed, but writing it explicitly documents the intent and
    // survives someone later changing a declaration.
    wire signed [ACC_W-1:0] prod_ext = {{(ACC_W-`ACCEL_PROD_W){prod[`ACCEL_PROD_W-1]}}, prod};

    always @(posedge clk) begin
        if (!resetn) begin
            acc       <= {ACC_W{1'b0}};
            mac_valid <= 1'b0;
        end else begin
            // Priority matters: clr wins over en. If both arrive in the same
            // cycle we clear and do NOT accumulate, so the first MAC of a
            // new output tile is never contaminated by the previous tile.
            if (clr) begin
                acc       <= {ACC_W{1'b0}};
                mac_valid <= 1'b0;
            end else if (en) begin
                acc       <= acc + prod_ext;
                mac_valid <= 1'b1;
            end else begin
                // Not enabled: HOLD the accumulator. Note there is no "else
                // acc <= 0" here. Writing one would be the classic beginner
                // bug that resets the sum every time the pipeline stalls.
                mac_valid <= 1'b0;
            end
        end
    end

endmodule
