//===========================================================================
// tb_mac_array.v -- Testbench for the parameterized MAC array
//===========================================================================
//
// The array claims to compute C[M][N] = A[M][K] * B[K][N] for one tile.
// This testbench checks the WHOLE result matrix against an independently
// computed reference, not just one element. That matters: an indexing bug in
// the flat-vector packing typically transposes the matrix or shifts it by
// one, and spot-checking element [0][0] -- which is often still correct --
// would miss it entirely.
//
// It also verifies the two things most likely to be wrong:
//   * signed arithmetic (negative operands, and -128 which has no positive
//     counterpart in int8)
//   * the clear/enable contract: accumulators must HOLD when en is low, and
//     must be wiped by clr so consecutive tiles do not contaminate each other
//
// The array geometry is a parameter of the testbench too, so the same file
// checks every configuration the sweep builds.
//
// RUN IT:
//   ./sim/run_icarus.sh tb_mac_array
//===========================================================================

`timescale 1ns / 1ps
`include "accel_pkg.vh"

module tb_mac_array;

    // Overridable from the command line so the sweep can check every shape:
    //   iverilog -Ptb_mac_array.H=8 -Ptb_mac_array.W=8 ...
    parameter H = 4;
    parameter W = 4;
    parameter K = 12;
    parameter ACC = `ACCEL_ACC_W;

    reg clk = 0, resetn = 0, clr = 0, en = 0;
    reg  [H*8-1:0] a_in = 0;
    reg  [W*8-1:0] b_in = 0;
    wire [H*W*ACC-1:0] acc_flat;
    wire [15:0] macs_this_cycle;

    integer errors = 0, tests = 0;

    always #5 clk = ~clk;

    mac_array #(.ARRAY_H(H), .ARRAY_W(W), .ACC_W(ACC), .PRECISION(8)) dut (
        .clk(clk), .resetn(resetn), .clr(clr), .en(en),
        .a_in(a_in), .b_in(b_in),
        .acc_flat(acc_flat), .macs_this_cycle(macs_this_cycle)
    );

    // Reference data and expected results, held in ordinary 2-D arrays --
    // something the DUT cannot use because Verilog-2001 forbids 2-D ports.
    // That difference is what makes this an independent check.
    reg signed [7:0]  A [0:H-1][0:K-1];
    reg signed [7:0]  B [0:K-1][0:W-1];
    reg signed [31:0] REF [0:H-1][0:W-1];

    integer i, j, k, trial;
    reg signed [31:0] got;

    task compute_reference;
        begin
            for (i = 0; i < H; i = i + 1)
                for (j = 0; j < W; j = j + 1) begin
                    REF[i][j] = 0;
                    for (k = 0; k < K; k = k + 1)
                        REF[i][j] = REF[i][j] + A[i][k] * B[k][j];
                end
        end
    endtask

    task run_tile;
        begin
            @(negedge clk); clr = 1'b1;
            @(negedge clk); clr = 1'b0;
            for (k = 0; k < K; k = k + 1) begin
                for (i = 0; i < H; i = i + 1) a_in[i*8 +: 8] = A[i][k];
                for (j = 0; j < W; j = j + 1) b_in[j*8 +: 8] = B[k][j];
                en = 1'b1;
                @(negedge clk);
            end
            en = 1'b0;
            @(negedge clk);
        end
    endtask

    task check_tile;
        input [255:0] name;
        begin
            for (i = 0; i < H; i = i + 1)
                for (j = 0; j < W; j = j + 1) begin
                    tests = tests + 1;
                    got = acc_flat[(i*W + j)*ACC +: ACC];
                    if (got !== REF[i][j]) begin
                        $display("FAIL %0s [%0d][%0d]: expected %0d, got %0d",
                                 name, i, j, REF[i][j], got);
                        errors = errors + 1;
                    end
                end
        end
    endtask

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_mac_array.vcd");
            $dumpvars(0, tb_mac_array);
        end

        @(posedge clk); resetn <= 1'b1; @(posedge clk);

        //-------------------------------------------------------------------
        // 1. All +1 -- the simplest possible sanity check. Every output
        //    should equal K. If this fails, the reduction depth is wrong.
        //-------------------------------------------------------------------
        for (i = 0; i < H; i = i + 1) for (k = 0; k < K; k = k + 1) A[i][k] = 8'sd1;
        for (k = 0; k < K; k = k + 1) for (j = 0; j < W; j = j + 1) B[k][j] = 8'sd1;
        compute_reference; run_tile; check_tile("all ones");

        //-------------------------------------------------------------------
        // 2. The int8 extreme: -128 * -128 accumulated K times. This is where
        //    a too-narrow accumulator or a lost sign bit shows up.
        //-------------------------------------------------------------------
        for (i = 0; i < H; i = i + 1) for (k = 0; k < K; k = k + 1) A[i][k] = -8'sd128;
        for (k = 0; k < K; k = k + 1) for (j = 0; j < W; j = j + 1) B[k][j] = -8'sd128;
        compute_reference; run_tile; check_tile("all -128");

        //-------------------------------------------------------------------
        // 3. Distinct values per row and column -- catches transposition and
        //    off-by-one in the flat packing, which "all ones" cannot.
        //-------------------------------------------------------------------
        for (i = 0; i < H; i = i + 1)
            for (k = 0; k < K; k = k + 1) A[i][k] = i + 1;
        for (k = 0; k < K; k = k + 1)
            for (j = 0; j < W; j = j + 1) B[k][j] = (j + 1) * 2;
        compute_reference; run_tile; check_tile("row/col identifiable");

        //-------------------------------------------------------------------
        // 4. Randomised trials, including negatives
        //-------------------------------------------------------------------
        for (trial = 0; trial < 20; trial = trial + 1) begin
            for (i = 0; i < H; i = i + 1)
                for (k = 0; k < K; k = k + 1) A[i][k] = $random;
            for (k = 0; k < K; k = k + 1)
                for (j = 0; j < W; j = j + 1) B[k][j] = $random;
            compute_reference; run_tile; check_tile("random");
        end

        //-------------------------------------------------------------------
        // 5. The enable contract: with en low the accumulators must HOLD.
        //    A design that zeroed them on !en would pass every test above
        //    and then fail the moment the controller stalls.
        //-------------------------------------------------------------------
        en = 1'b0;
        a_in = {(H*8){1'b1}};   // drive garbage on the operand buses
        b_in = {(W*8){1'b1}};
        repeat (5) @(negedge clk);
        check_tile("accumulators hold while en is low");

        //-------------------------------------------------------------------
        // 6. clr must wipe every accumulator.
        //-------------------------------------------------------------------
        @(negedge clk); clr = 1'b1;
        @(negedge clk); clr = 1'b0;
        @(negedge clk);
        for (i = 0; i < H; i = i + 1)
            for (j = 0; j < W; j = j + 1) begin
                tests = tests + 1;
                got = acc_flat[(i*W + j)*ACC +: ACC];
                if (got !== 32'sd0) begin
                    $display("FAIL clr [%0d][%0d]: expected 0, got %0d", i, j, got);
                    errors = errors + 1;
                end
            end

        $display("");
        $display("=====================================================");
        $display(" tb_mac_array %0dx%0d K=%0d: %0d checks, %0d failures",
                 H, W, K, tests, errors);
        $display("=====================================================");
        if (errors == 0) $display("TEST PASSED");
        else             $display("TEST FAILED");
        $finish;
    end

    initial begin
        #5_000_000;
        $display("TEST FAILED -- global timeout");
        $finish;
    end

endmodule
