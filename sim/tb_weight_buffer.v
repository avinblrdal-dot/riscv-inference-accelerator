//===========================================================================
// tb_weight_buffer.v -- Testbench for local weight storage
//===========================================================================
//
// Two configurations are checked side by side in the SAME simulation:
//
//   DEPTH = 256  the real buffer -- data written must come back intact
//   DEPTH = 0    the CONTROL CASE -- no storage, reads pass through
//
// Testing them together matters scientifically. The DEPTH=0 build is the
// experimental control for RQ3, so it has to be a REAL, working
// configuration -- not a broken one that happens to compile. If DEPTH=0 were
// subtly non-functional, every "buffering helps" result would be an artifact
// of comparing against a broken baseline rather than a fair one.
//
// RUN IT:
//   ./sim/run_icarus.sh tb_weight_buffer
//===========================================================================

`timescale 1ns / 1ps
`include "accel_pkg.vh"

module tb_weight_buffer;

    localparam DEPTH = 256;

    reg clk = 0, resetn = 0;
    reg wr_en = 0, rd_en = 0;
    reg [15:0] wr_addr = 0, rd_addr = 0;
    reg [31:0] wr_data = 0, bypass = 0;

    wire [31:0] rd_data_buf, rd_data_nobuf;
    wire [31:0] fill_buf, fill_nobuf;

    integer errors = 0, tests = 0;

    always #5 clk = ~clk;

    weight_buffer #(.DEPTH(DEPTH), .WIDTH(32), .PRECISION(8)) u_buf (
        .clk(clk), .resetn(resetn),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_data_buf),
        .bypass_data(bypass), .fill_count(fill_buf)
    );

    weight_buffer #(.DEPTH(0), .WIDTH(32), .PRECISION(8)) u_nobuf (
        .clk(clk), .resetn(resetn),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_data_nobuf),
        .bypass_data(bypass), .fill_count(fill_nobuf)
    );

    integer i;
    reg [31:0] expect_word;

    task check32;
        input [31:0] got, exp;
        input [255:0] name;
        begin
            tests = tests + 1;
            if (got !== exp) begin
                $display("FAIL %0s: expected %08x, got %08x", name, exp, got);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_weight_buffer.vcd");
            $dumpvars(0, tb_weight_buffer);
        end

        @(posedge clk); resetn <= 1'b1; @(posedge clk);

        //-------------------------------------------------------------------
        // 1. Write a recognisable pattern across the whole buffer.
        //-------------------------------------------------------------------
        // The pattern encodes the address, so a read returning the wrong
        // word tells you immediately WHICH word it returned -- far more
        // useful than a pass/fail when debugging an addressing bug.
        for (i = 0; i < DEPTH; i = i + 1) begin
            @(negedge clk);
            wr_en   = 1'b1;
            wr_addr = i[15:0];
            wr_data = {16'hBEEF, i[15:0]};
        end
        @(negedge clk); wr_en = 1'b0;

        //-------------------------------------------------------------------
        // 2. Read it all back. Remember the ONE CYCLE of read latency: the
        //    address is registered, and data appears the following cycle.
        //    Forgetting this is the most common way to "prove" a working
        //    memory is broken.
        //-------------------------------------------------------------------
        for (i = 0; i < DEPTH; i = i + 1) begin
            @(negedge clk);
            rd_en   = 1'b1;
            rd_addr = i[15:0];
            @(negedge clk);          // latency cycle
            expect_word = {16'hBEEF, i[15:0]};
            check32(rd_data_buf, expect_word, "buffered readback");
        end
        @(negedge clk); rd_en = 1'b0;

        //-------------------------------------------------------------------
        // 3. fill_count must report DISTINCT entries written, not the number
        //    of write operations. Rewriting an address must not inflate it,
        //    because the sweep reports it as achieved reuse capacity.
        //-------------------------------------------------------------------
        check32(fill_buf, DEPTH, "fill_count after filling every entry");

        @(negedge clk); wr_en = 1'b1; wr_addr = 16'd7; wr_data = 32'hAAAA5555;
        @(negedge clk); wr_en = 1'b0;
        check32(fill_buf, DEPTH, "fill_count unchanged when rewriting an entry");

        @(negedge clk); rd_en = 1'b1; rd_addr = 16'd7;
        @(negedge clk);
        check32(rd_data_buf, 32'hAAAA5555, "overwrite takes effect");
        @(negedge clk); rd_en = 1'b0;

        //-------------------------------------------------------------------
        // 4. The DEPTH=0 control case.
        //-------------------------------------------------------------------
        // It must pass bypass_data straight through, and must report zero
        // reuse -- that is the definition of the control.
        @(negedge clk);
        bypass = 32'h12345678;
        rd_en  = 1'b1;
        rd_addr = 16'd42;
        @(negedge clk);
        check32(rd_data_nobuf, 32'h12345678, "DEPTH=0 passes bypass through");
        check32(fill_nobuf, 32'd0, "DEPTH=0 reports zero reuse");

        @(negedge clk);
        bypass = 32'hCAFEF00D;
        @(negedge clk);
        check32(rd_data_nobuf, 32'hCAFEF00D, "DEPTH=0 tracks bypass with no storage");

        $display("");
        $display("=====================================================");
        $display(" tb_weight_buffer: %0d checks, %0d failures", tests, errors);
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
