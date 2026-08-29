`timescale 1ns / 1ps
//===========================================================================
// mac_array.v -- Parameterized ARRAY_H x ARRAY_W grid of MAC cells
//===========================================================================
//
// WHAT IT COMPUTES:
//
//   A small matrix multiply tile.  C[i][j] = sum over k of A[i][k]*B[k][j]
//   for i in 0..ARRAY_H-1 and j in 0..ARRAY_W-1.
//
//   Every neural network layer we care about reduces to matrix multiply:
//   a fully-connected layer is one directly, and a convolution becomes one
//   after "im2col" (unfolding each sliding window into a column). So this
//   one block accelerates both workloads.
//
// THE DATAFLOW -- output stationary broadcast:
//
//   Each cell owns ONE output element C[i][j] and keeps its accumulator
//   there for the whole reduction (hence "output stationary"). On each
//   cycle we broadcast one column of A and one row of B:
//
//                    b_in[0]  b_in[1]  b_in[2]   <-- weights, one per column
//                       |        |        |
//        a_in[0] ----[cell00][cell01][cell02]
//        a_in[1] ----[cell10][cell11][cell12]
//                       ^
//                       |  each cell multiplies its row's a by its column's b
//
//   After K cycles every cell holds a finished dot product.
//
// WHY THIS SHAPE IS THE WHOLE ARGUMENT (RQ3):
//
//   Look at the wire count. Per cycle this array consumes
//       ARRAY_H + ARRAY_W  bytes of input
//   and performs
//       ARRAY_H * ARRAY_W  multiply-accumulates.
//
//   The ratio -- MACs per byte fetched -- is called ARITHMETIC INTENSITY:
//
//       intensity = (H*W) / (H+W)
//
//       1x1 array:  1/2  = 0.5 MACs per byte
//       2x2 array:  4/4  = 1.0
//       4x4 array: 16/8  = 2.0
//       8x8 array: 64/16 = 4.0
//
//   So a bigger array is not just more compute -- it is intrinsically more
//   EFFICIENT per byte moved, because each byte fetched gets used by a whole
//   row or column of cells instead of by one. This is the "data reuse"
//   effect, and it is why we predict a buffer x array-width INTERACTION
//   rather than two independent main effects: the wider the array, the more
//   there is to gain from feeding it well, and the more it starves if you
//   do not. analysis/anova.py tests exactly that interaction term.
//
//   The broadcast also has a cost: a signal fanning out to 8 cells has more
//   capacitance and a longer critical path than one going to 2. Expect Fmax
//   to fall as ARRAY_W grows. That trade-off is real and should appear in
//   the timing column of the sweep -- if it does not, be suspicious that
//   the parameter is not actually propagating.
//
// PARAMETERS:
//   ARRAY_H, ARRAY_W   grid dimensions (1,2,4,8 in the sweep)
//   ACC_W              accumulator width per cell
//   PRECISION          8 or 4
//
// PORTS:
//   clr        clear all accumulators -- start a new output tile
//   en         perform one MAC step across the whole array this cycle
//   a_in       ARRAY_H activation bytes, packed into one flat vector
//   b_in       ARRAY_W weight bytes, packed into one flat vector
//   acc_flat   all H*W accumulators, packed flat
//   macs_this_cycle  how many real MACs happened (0 or H*W). Feeds the
//                    perf counter so we can separate useful work from stalls.
//
// A NOTE ON "PACKED FLAT VECTORS":
//   Verilog-2001 cannot pass a 2-D array through a module port. The standard
//   workaround is to flatten: a_in is ARRAY_H*8 bits wide and element i is
//   a_in[i*8 +: 8]. The "+:" is an indexed part-select and means "starting
//   at bit i*8, take 8 bits going up". This is ugly but universally
//   supported; SystemVerilog would let us write it properly, but Vivado's
//   Verilog-2001 path is the most reliable for synthesis.
//===========================================================================

`include "accel_pkg.vh"

module mac_array #(
    parameter ARRAY_H   = 4,
    parameter ARRAY_W   = 4,
    parameter ACC_W     = `ACCEL_ACC_W,
    parameter PRECISION = 8
) (
    input  wire                          clk,
    input  wire                          resetn,
    input  wire                          clr,
    input  wire                          en,
    input  wire [ARRAY_H*8-1:0]          a_in,
    input  wire [ARRAY_W*8-1:0]          b_in,
    output wire [ARRAY_H*ARRAY_W*ACC_W-1:0] acc_flat,
    output reg  [15:0]                   macs_this_cycle
);

    genvar gi, gj;

    generate
        for (gi = 0; gi < ARRAY_H; gi = gi + 1) begin : g_row
            for (gj = 0; gj < ARRAY_W; gj = gj + 1) begin : g_col

                // Slice this cell's operands out of the flat input vectors.
                // Cell (i,j) sees row i's activation and column j's weight.
                wire signed [7:0] a_cell = a_in[gi*8 +: 8];
                wire signed [7:0] b_cell = b_in[gj*8 +: 8];

                wire signed [ACC_W-1:0] acc_cell;
                wire                    v_cell;

                mac_unit #(
                    .ACC_W      (ACC_W),
                    .PRECISION  (PRECISION),
                    .WEIGHT_REG (0)          // output-stationary
                ) u_mac (
                    .clk       (clk),
                    .resetn    (resetn),
                    .clr       (clr),
                    .en        (en),
                    .load_w    (1'b0),       // unused when WEIGHT_REG=0
                    .a_in      (a_cell),
                    .b_in      (b_cell),
                    .acc_out   (acc_cell),
                    .mac_valid (v_cell)
                );

                // Pack this cell's accumulator into the flat output at its
                // row-major position. Getting this index expression wrong is
                // an easy off-by-one that shows up as a transposed or
                // shifted result matrix -- sim/tb_mac_array.v checks the
                // full matrix against a reference, not just one element,
                // specifically to catch that.
                assign acc_flat[(gi*ARRAY_W + gj)*ACC_W +: ACC_W] = acc_cell;
            end
        end
    endgenerate

    //-----------------------------------------------------------------------
    // Useful-work counter
    //-----------------------------------------------------------------------
    // Every cell is enabled together, so either all H*W cells did a MAC this
    // cycle or none did. We report the count rather than a flag so that
    // perf_counter.v can accumulate total MACs directly and the analysis can
    // compute utilisation = MACs_done / (H*W * total_cycles).
    always @(posedge clk) begin
        if (!resetn)      macs_this_cycle <= 16'd0;
        else if (clr)     macs_this_cycle <= 16'd0;
        else if (en)      macs_this_cycle <= ARRAY_H * ARRAY_W;
        else              macs_this_cycle <= 16'd0;
    end

    // Report the built geometry once at time 0. When the sweep runs 48
    // configurations, this line in the log is how you confirm the parameter
    // actually reached the RTL instead of silently defaulting.
    initial begin
        $display("[mac_array] built %0dx%0d PRECISION=%0d -> %0d MACs/cycle, arithmetic intensity %0d/%0d",
                 ARRAY_H, ARRAY_W, PRECISION, ARRAY_H*ARRAY_W,
                 ARRAY_H*ARRAY_W, ARRAY_H+ARRAY_W);
    end

endmodule
