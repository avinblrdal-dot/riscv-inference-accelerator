`timescale 1ns / 1ps
//===========================================================================
// weight_buffer.v -- Local on-chip storage for weights
//===========================================================================
//
// WHY THIS MODULE EXISTS -- this is research question RQ3
//
//   The obvious way to make an accelerator faster is to add more
//   multipliers. That intuition is usually wrong, and this buffer is how we
//   test it.
//
//   Consider an 8x8 array: 64 multipliers, each wanting one weight byte and
//   one activation byte per cycle = 128 bytes/cycle. Our memory bus is 32
//   bits = 4 bytes/cycle. So the array can be fed at 3% of its appetite.
//   The other 97% of cycles the multipliers sit idle waiting for data. You
//   have paid for 64 multipliers and are getting the throughput of two.
//
//   The fix is not more compute, it is DATA REUSE. Weights in a neural
//   network layer are used over and over across many input positions. If we
//   copy a tile of weights into fast local storage once, we can then read it
//   thousands of times without touching main memory. Main memory access also
//   costs far more ENERGY than local SRAM access -- often 10-100x -- so this
//   is an energy argument even more than a speed argument.
//
//   DEPTH=0 is the deliberate CONTROL CASE: no local buffer at all, every
//   operand fetched from main memory. Comparing DEPTH=0 against DEPTH=1024
//   at each array width is exactly the interaction effect RQ3 asks about:
//   does buffering matter MORE when the array is wider? (We predict yes,
//   strongly.) analysis/anova.py tests that interaction term directly.
//
// HOW A BEGINNER SHOULD READ THIS:
//
//   An FPGA has "block RAM" (BRAM): small, fast memories built into the
//   chip. In Verilog you do not instantiate them by name; you write an array
//   and a specific access pattern, and the synthesis tool RECOGNISES the
//   pattern and maps it onto BRAM. That recognition is fragile -- change the
//   style and you get thousands of flip-flops instead, blowing the area
//   budget. The style below (synchronous read, registered address) is the
//   one Vivado reliably infers as BRAM.
//
// PARAMETERS:
//   DEPTH      number of entries. 0 means "no buffer" (control case).
//   WIDTH      bits per entry (32 = four packed int8 lanes).
//   PRECISION  8 or 4. At 4, entries hold twice as many values (8 nibbles
//              per 32-bit word), so the same DEPTH buys twice the reuse.
//              This is a real effect the sweep should see.
//
// PORTS:
//   wr_en/wr_addr/wr_data   write port (filled by the CPU or a DMA)
//   rd_en/rd_addr/rd_data   read port, ONE CYCLE of latency
//   bypass_data             used when DEPTH=0: data comes straight through
//   full/empty-ish status is deliberately absent; the controller tracks
//   occupancy itself, because it knows the tile geometry.
//===========================================================================

`include "accel_pkg.vh"

module weight_buffer #(
    parameter DEPTH     = 256,
    parameter WIDTH     = 32,
    parameter PRECISION = 8
) (
    input  wire             clk,
    input  wire             resetn,

    // write port
    input  wire             wr_en,
    input  wire [15:0]      wr_addr,
    input  wire [WIDTH-1:0] wr_data,

    // read port (1-cycle latency when DEPTH > 0)
    input  wire             rd_en,
    input  wire [15:0]      rd_addr,
    output wire [WIDTH-1:0] rd_data,

    // used only when DEPTH == 0
    input  wire [WIDTH-1:0] bypass_data,

    // observability: how many distinct entries have been written since reset.
    // Used by the perf counters to report achieved reuse.
    output reg  [31:0]      fill_count
);

    // ADDR_W: how many address bits DEPTH actually needs. clog2 of 256 is 8.
    // Guard against DEPTH=0 (clog2(0) is undefined) and DEPTH=1.
    localparam ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

    generate
        if (DEPTH == 0) begin : g_nobuf
            //-------------------------------------------------------------
            // CONTROL CASE: no storage at all.
            //
            // Every read returns whatever main memory happens to be
            // presenting this cycle. There is no reuse, so the controller
            // must re-fetch an operand every single time it is needed. This
            // configuration should be markedly slower AND markedly less
            // energy-efficient; if the sweep does NOT show that, something
            // is wrong with the experiment, not with the theory.
            //
            // We still declare the module ports identically so that nothing
            // upstream has to know which variant was built. That is what
            // makes the sweep a fair comparison.
            //-------------------------------------------------------------
            assign rd_data = bypass_data;

            always @(posedge clk) begin
                if (!resetn) fill_count <= 32'd0;
                // fill_count stays 0: by definition nothing is ever buffered.
            end

            // Silence "unused input" lint warnings without changing behaviour.
            wire _unused = &{1'b0, wr_en, wr_addr, wr_data, rd_en, rd_addr, 1'b0};

        end else begin : g_buf
            //-------------------------------------------------------------
            // REAL BUFFER: a simple dual-port synchronous RAM.
            //-------------------------------------------------------------
            // This declaration -- a reg array indexed by an address -- is
            // how you describe a memory in Verilog. There is no "malloc";
            // the size is fixed at compile time.
            reg [WIDTH-1:0] mem [0:DEPTH-1];

            // Tracks which entries have been written, so fill_count is a
            // count of DISTINCT entries rather than of write operations.
            reg             written [0:DEPTH-1];

            reg [WIDTH-1:0] rd_data_q;
            assign rd_data = rd_data_q;

            integer i;
            always @(posedge clk) begin
                if (!resetn) begin
                    fill_count <= 32'd0;
                    rd_data_q  <= {WIDTH{1'b0}};
                    // Clearing the "written" flags on reset is what makes
                    // fill_count meaningful across back-to-back sweep runs.
                    // We deliberately do NOT clear mem[] here: initialising
                    // a large array in a reset branch prevents Vivado from
                    // inferring BRAM and would silently explode the area
                    // numbers we are trying to measure.
                    for (i = 0; i < DEPTH; i = i + 1)
                        written[i] <= 1'b0;
                end else begin
                    if (wr_en) begin
                        mem[wr_addr[ADDR_W-1:0]] <= wr_data;
                        if (!written[wr_addr[ADDR_W-1:0]]) begin
                            written[wr_addr[ADDR_W-1:0]] <= 1'b1;
                            fill_count <= fill_count + 32'd1;
                        end
                    end
                    // Synchronous read: the address is registered and data
                    // appears on the NEXT cycle. Every consumer of rd_data
                    // must account for that one-cycle delay -- accel_ctrl.v
                    // does this explicitly in its address pipeline.
                    if (rd_en)
                        rd_data_q <= mem[rd_addr[ADDR_W-1:0]];
                end
            end

            wire _unused = &{1'b0, bypass_data, 1'b0};
        end
    endgenerate

    //-----------------------------------------------------------------------
    // Effective capacity in VALUES (not words), for documentation and for
    // the sweep metadata. At PRECISION=4 a 32-bit word holds 8 values
    // instead of 4, so the same DEPTH gives twice the reuse.
    //-----------------------------------------------------------------------
    localparam VALUES_PER_WORD = (PRECISION == 4) ? (WIDTH / 4) : (WIDTH / 8);
    localparam CAPACITY_VALUES = DEPTH * VALUES_PER_WORD;

    initial begin
        if (DEPTH == 0)
            $display("[weight_buffer] DEPTH=0 -- CONTROL CASE, no local reuse");
        else
            $display("[weight_buffer] DEPTH=%0d WIDTH=%0d PRECISION=%0d -> capacity %0d values",
                     DEPTH, WIDTH, PRECISION, CAPACITY_VALUES);
    end

endmodule
