//===========================================================================
// tb_dot4.v -- Standalone testbench for the PCPI custom-instruction unit
//===========================================================================
//
// WHAT A TESTBENCH IS:
//
//   A testbench is not hardware. It is a Verilog module that never gets
//   synthesised -- it exists only inside the simulator. Its job is to
//   generate a clock, wiggle the inputs of the module under test ("DUT" =
//   Device Under Test), and check the outputs against values it computes
//   independently.
//
//   The golden rule: the testbench must compute the expected answer a
//   DIFFERENT WAY than the DUT does. If you check the design against itself
//   you will prove only that it is self-consistent, not that it is correct.
//   Here the reference is a plain behavioural expression using Verilog's own
//   signed arithmetic, which is a genuinely independent path from the DUT's
//   explicit lane unpacking.
//
// RUN IT:
//   ./sim/run_icarus.sh tb_dot4
//
// WHAT IT COVERS:
//   1. DOT4 over directed edge cases (zeros, all -128, all +127, mixed sign)
//   2. DOT4 over randomised vectors against a behavioural reference
//   3. DOT4A accumulation across many instructions
//   4. ACCRD read-and-clear semantics
//   5. REGRESSION: valid held high for many cycles must apply the
//      instruction EXACTLY ONCE  (see docs/DECISIONS.md, "PCPI one-shot")
//   6. DEADLOCK GUARD: a foreign instruction must get neither wait nor ready
//   7. DEADLOCK GUARD: every claimed instruction must assert ready within a
//      bounded number of cycles
//===========================================================================

`timescale 1ns / 1ps
`include "accel_pkg.vh"

module tb_dot4;

    reg         clk = 0;
    reg         resetn = 0;
    reg         pcpi_valid = 0;
    reg  [31:0] pcpi_insn = 0, pcpi_rs1 = 0, pcpi_rs2 = 0;
    wire        pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_rd, acc_value;

    integer errors = 0;
    integer tests  = 0;

    // Where the result is captured on the cycle pcpi_ready is high.
    //
    // This matters and is easy to get wrong: pcpi_rd is only meaningful
    // DURING the ready cycle. dot4_pcpi.v deliberately drives it back to
    // zero afterwards so a stale value can never be mistaken for a fresh
    // one. The real CPU latches rd exactly when ready is high, so the
    // testbench must do the same -- checking pcpi_rd after the handshake
    // reads zero and every test appears to fail.
    reg signed [31:0] captured_rd;
    reg               captured_wr;

    // 100 MHz: 10 ns period.
    always #5 clk = ~clk;

    dot4_pcpi #(.PIPELINE(0), .PRECISION(8)) dut (
        .clk(clk), .resetn(resetn),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
        .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .acc_value(acc_value), .acc_active()
    );

    // Build an R-type custom-0 instruction word with the given funct7.
    function [31:0] mk_insn;
        input [6:0] f7;
        begin
            mk_insn = {f7, 5'd6, 5'd5, `ACCEL_FUNCT3_DOT, 5'd7,
                       `ACCEL_OPCODE_CUSTOM0};
        end
    endfunction

    // The INDEPENDENT reference model: unpack four signed bytes and sum the
    // products, using Verilog's own signed arithmetic rather than the DUT's
    // structure.
    function signed [31:0] ref_dot4;
        input [31:0] a, b;
        reg signed [7:0] a0, a1, a2, a3, b0, b1, b2, b3;
        begin
            a0 = a[7:0];   a1 = a[15:8];  a2 = a[23:16]; a3 = a[31:24];
            b0 = b[7:0];   b1 = b[15:8];  b2 = b[23:16]; b3 = b[31:24];
            ref_dot4 = (a0 * b0) + (a1 * b1) + (a2 * b2) + (a3 * b3);
        end
    endfunction

    // Issue one instruction the way the real core does: raise valid and hold
    // it until ready comes back. ``hold_extra`` keeps valid high for extra
    // cycles afterwards, which is what test 5 uses to catch double-execution.
    task issue;
        input [6:0]  f7;
        input [31:0] a, b;
        input integer hold_extra;
        integer guard, i;
        begin
            @(negedge clk);
            pcpi_insn  = mk_insn(f7);
            pcpi_rs1   = a;
            pcpi_rs2   = b;
            pcpi_valid = 1'b1;

            // DEADLOCK GUARD. If the DUT never asserts ready, the real CPU
            // would hang forever with no error message -- the simulation
            // just goes quiet. Bounding the wait converts that silent hang
            // into a loud, diagnosable failure.
            guard       = 0;
            captured_rd = 32'sd0;
            captured_wr = 1'b0;
            while (!pcpi_ready && guard < 64) begin
                @(posedge clk);
                #1;
                guard = guard + 1;
            end
            // Sample on the ready cycle, exactly as the core does.
            if (pcpi_ready) begin
                captured_rd = pcpi_rd;
                captured_wr = pcpi_wr;
            end
            if (guard >= 64) begin
                $display("FATAL: no pcpi_ready within 64 cycles for funct7=%b -- the core would have DEADLOCKED here", f7);
                errors = errors + 1;
            end

            for (i = 0; i < hold_extra; i = i + 1) @(posedge clk);

            @(negedge clk);
            pcpi_valid = 1'b0;
            pcpi_insn  = 32'd0;
            @(negedge clk);
        end
    endtask

    task check32;
        input signed [31:0] got, exp;
        input [255:0] name;
        begin
            tests = tests + 1;
            if (got !== exp) begin
                $display("FAIL %0s: expected %0d, got %0d", name, exp, got);
                errors = errors + 1;
            end
        end
    endtask

    integer i;
    reg [31:0] ra, rb;
    reg signed [31:0] expect_acc;

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_dot4.vcd");
            $dumpvars(0, tb_dot4);
        end

        @(posedge clk);
        resetn <= 1'b1;
        @(posedge clk);

        //-------------------------------------------------------------------
        // 1. Directed edge cases
        //-------------------------------------------------------------------
        // All zeros.
        issue(`ACCEL_FUNCT7_DOT4, 32'h00000000, 32'h00000000, 0);
        check32(captured_rd, 32'sd0, "DOT4 zeros");

        // The most-negative int8 squared, four times: 4 * (-128 * -128)
        // = 4 * 16384 = 65536. This is the case that overflows if anyone
        // ever narrows the accumulator to 16 bits.
        issue(`ACCEL_FUNCT7_DOT4, 32'h80808080, 32'h80808080, 0);
        check32(captured_rd, 32'sd65536, "DOT4 all -128 squared");

        // Largest positive: 4 * (127 * 127) = 64516.
        issue(`ACCEL_FUNCT7_DOT4, 32'h7F7F7F7F, 32'h7F7F7F7F, 0);
        check32(captured_rd, 32'sd64516, "DOT4 all +127 squared");

        // Mixed signs -- the case that fails if $signed() is ever dropped.
        issue(`ACCEL_FUNCT7_DOT4, 32'h01020304, 32'hFF01FF01, 0);
        check32(captured_rd, 32'sd2, "DOT4 mixed sign");

        // A single -1 lane against +1: proves sign extension of one lane
        // does not leak into its neighbours.
        issue(`ACCEL_FUNCT7_DOT4, 32'h000000FF, 32'h00000001, 0);
        check32(captured_rd, -32'sd1, "DOT4 single negative lane");

        //-------------------------------------------------------------------
        // 2. Randomised comparison against the reference model
        //-------------------------------------------------------------------
        for (i = 0; i < 200; i = i + 1) begin
            ra = $random;
            rb = $random;
            issue(`ACCEL_FUNCT7_DOT4, ra, rb, 0);
            check32(captured_rd, ref_dot4(ra, rb), "DOT4 random");
        end

        //-------------------------------------------------------------------
        // 3 & 4. Accumulate, then read-and-clear
        //-------------------------------------------------------------------
        issue(`ACCEL_FUNCT7_ACCRD, 0, 0, 0);   // start from a known state
        expect_acc = 0;
        for (i = 0; i < 32; i = i + 1) begin
            ra = $random;
            rb = $random;
            expect_acc = expect_acc + ref_dot4(ra, rb);
            issue(`ACCEL_FUNCT7_DOT4A, ra, rb, 0);
        end
        issue(`ACCEL_FUNCT7_ACCRD, 0, 0, 0);
        check32(captured_rd, expect_acc, "DOT4A accumulate over 32 instructions");

        issue(`ACCEL_FUNCT7_ACCRD, 0, 0, 0);
        check32(captured_rd, 32'sd0, "ACCRD cleared the accumulator");

        //-------------------------------------------------------------------
        // 5. REGRESSION: one instruction must take effect exactly once
        //-------------------------------------------------------------------
        // PicoRV32 holds pcpi_valid high until it sees pcpi_ready, which is
        // at least one cycle after we answer. An earlier version of
        // dot4_pcpi.v accumulated on EVERY cycle valid was high, so a single
        // DOT4A added its product twice and every dot product came out at 2x.
        // The bug was invisible in a testbench that dropped valid promptly --
        // hence this test, which deliberately holds it for many cycles.
        issue(`ACCEL_FUNCT7_ACCRD, 0, 0, 0);        // clear
        issue(`ACCEL_FUNCT7_DOT4A, 32'h01010101, 32'h02020202, 8);
        // 4 lanes * (1*2) = 8, applied ONCE regardless of how long valid held.
        issue(`ACCEL_FUNCT7_ACCRD, 0, 0, 0);
        check32(captured_rd, 32'sd8, "DOT4A applied exactly once with valid held 8 cycles");

        //-------------------------------------------------------------------
        // 6. A foreign instruction must be completely ignored
        //-------------------------------------------------------------------
        // If we claimed instructions that are not ours, the core could never
        // trap a genuinely illegal instruction. If we asserted wait without
        // ready, the core would hang.
        @(negedge clk);
        pcpi_insn  = 32'h00000033;   // a plain ADD -- not ours
        pcpi_valid = 1'b1;
        @(posedge clk);
        #1;
        tests = tests + 1;
        if (pcpi_ready !== 1'b0 || pcpi_wait !== 1'b0) begin
            $display("FAIL foreign instruction: ready=%b wait=%b (both must be 0)", pcpi_ready, pcpi_wait);
            errors = errors + 1;
        end
        @(negedge clk);
        pcpi_valid = 1'b0;

        //-------------------------------------------------------------------
        $display("");
        $display("=====================================================");
        $display(" tb_dot4: %0d checks, %0d failures", tests, errors);
        $display("=====================================================");
        if (errors == 0) $display("TEST PASSED");
        else             $display("TEST FAILED");
        $finish;
    end

    // Global watchdog: if the whole testbench wedges, say so rather than
    // letting CI sit until it times out with no explanation.
    initial begin
        #2_000_000;
        $display("TEST FAILED -- global timeout, something is deadlocked");
        $finish;
    end

endmodule
