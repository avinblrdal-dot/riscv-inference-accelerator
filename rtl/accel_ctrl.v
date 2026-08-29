`timescale 1ns / 1ps
//===========================================================================
// accel_ctrl.v -- Sequencer (FSM) that drives the MAC array
//===========================================================================
//
// WHAT A FINITE STATE MACHINE IS, for a beginner:
//
//   The MAC array is pure arithmetic -- it has no idea what a matrix is. It
//   needs something to tell it, every cycle, "here are the next operands,
//   accumulate them", and eventually "you are done, hand over the results".
//   That something is a state machine: a register holding "where am I in the
//   job", plus rules for moving to the next state.
//
//   In C you would write three nested for-loops. In hardware you cannot --
//   there is no program counter here. So you build the loops explicitly out
//   of counters (k_cnt, tile_i, tile_j below) and a state register, and you
//   advance them by hand on every clock edge. That is genuinely all an FSM
//   is: the nested loops of the C version, turned inside out.
//
// THE JOB:  C[M][N] = A[M][K] * B[K][N], computed in tiles.
//
//   The array is only ARRAY_H x ARRAY_W, but M and N can be much larger. So
//   we chop the output into tiles of that size and do one tile at a time:
//
//     for tile_i in steps of ARRAY_H:
//       for tile_j in steps of ARRAY_W:
//         clear accumulators               <- state S_CLEAR
//         for k in 0..K-1:                 <- state S_STREAM
//           feed one column of A, one row of B
//         push ARRAY_H*ARRAY_W results out <- state S_DRAIN
//
// WHERE THE STALLS COME FROM (this is the RQ3 evidence path):
//
//   In S_STREAM we need ARRAY_H activation bytes and ARRAY_W weight bytes
//   EVERY cycle. Where they come from depends on how the design was built:
//
//     WBUF_DEPTH > 0 : operands are already in local BRAM. One cycle of read
//                      latency, fully pipelined -> no stall, en stays high.
//     WBUF_DEPTH = 0 : operands must be fetched from main memory over a
//                      32-bit bus, i.e. 4 bytes per cycle. An 8x8 array
//                      wants 16 bytes per cycle, so we can only issue a MAC
//                      step every ceil(16/4) = 4 cycles. The other 3 are
//                      STALLS, and we count every one of them.
//
//   That ratio is MEM_BEATS below. It is the mechanism by which buffer depth
//   and array width INTERACT: MEM_BEATS grows with array width only when
//   there is no buffer; with a buffer it stays 1 regardless of width. A
//   statistical interaction term is exactly the right tool to detect that,
//   which is why EXPERIMENT_PLAN.md specifies a factorial design rather than
//   changing one factor at a time.
//
// PARAMETERS: ARRAY_H, ARRAY_W, WBUF_DEPTH, ABUF_DEPTH, PRECISION, BUS_BYTES
//===========================================================================

`include "accel_pkg.vh"

module accel_ctrl #(
    parameter ARRAY_H    = 4,
    parameter ARRAY_W    = 4,
    parameter WBUF_DEPTH = 256,
    parameter ABUF_DEPTH = 256,
    parameter PRECISION  = 8,
    parameter BUS_BYTES  = 4     // main memory delivers this many bytes/cycle
) (
    input  wire        clk,
    input  wire        resetn,

    // ---- control interface (from the memory-mapped registers) -----------
    input  wire        start,        // one-cycle pulse
    input  wire [15:0] dim_m,
    input  wire [15:0] dim_n,
    input  wire [15:0] dim_k,
    output reg         busy,
    output reg         done,

    // ---- to the MAC array ------------------------------------------------
    output reg         arr_clr,
    output reg         arr_en,

    // ---- operand fetch ---------------------------------------------------
    output reg         wbuf_rd_en,
    output reg  [15:0] wbuf_rd_addr,
    output reg         abuf_rd_en,
    output reg  [15:0] abuf_rd_addr,

    // ---- result drain ----------------------------------------------------
    output reg         res_push,
    output reg  [15:0] res_tile_i,
    output reg  [15:0] res_tile_j,

    // ---- performance observability --------------------------------------
    output reg         stall_o
);

    //-----------------------------------------------------------------------
    // How many bus cycles one MAC step costs when there is NO local buffer.
    //-----------------------------------------------------------------------
    // Bytes needed per step = ARRAY_H (activations) + ARRAY_W (weights).
    // With a buffer all of that comes from BRAM in one pipelined cycle, so
    // the cost is 1. Without, it is ceil(bytes / BUS_BYTES).
    //
    // At PRECISION=4 two values share a byte, so half as many bytes move --
    // a real and separately measurable bandwidth advantage of low precision
    // that is easy to overlook when thinking only about multiplier size.
    localparam OPERAND_BYTES_8 = ARRAY_H + ARRAY_W;
    localparam OPERAND_BYTES   = (PRECISION == 4)
                                 ? ((OPERAND_BYTES_8 + 1) / 2)
                                 : OPERAND_BYTES_8;
    localparam BUFFERED        = (WBUF_DEPTH > 0) && (ABUF_DEPTH > 0);
    localparam MEM_BEATS       = BUFFERED
                                 ? 1
                                 : ((OPERAND_BYTES + BUS_BYTES - 1) / BUS_BYTES);

    //-----------------------------------------------------------------------
    // State encoding
    //-----------------------------------------------------------------------
    // localparam, not `define: these are local names. A `define here would
    // leak S_IDLE into every file that includes this one and eventually
    // collide with something else.
    localparam S_IDLE   = 3'd0,
               S_CLEAR  = 3'd1,
               S_STREAM = 3'd2,
               S_DRAIN  = 3'd3,
               S_NEXT   = 3'd4,
               S_DONE   = 3'd5;

    reg [2:0]  state;
    reg [15:0] k_cnt;
    reg [15:0] tile_i;
    reg [15:0] tile_j;
    reg [7:0]  beat;

    // Latched job dimensions. We copy them at start so software rewriting
    // the registers mid-run cannot corrupt an in-flight job -- a subtle bug
    // that would only ever appear under interrupts.
    reg [15:0] m_q, n_q, k_q;

    // Loop-boundary tests as named wires: burying these comparisons inside
    // the state logic is where off-by-one errors hide. Note >= rather than
    // ==: if software programs dim_k = 0, the == form would run 65536 times
    // before wrapping, while >= exits immediately.
    wire k_last    = (k_cnt + 16'd1) >= k_q;
    wire i_last    = (tile_i + ARRAY_H) >= m_q;
    wire j_last    = (tile_j + ARRAY_W) >= n_q;
    wire beat_last = (beat + 8'd1) >= MEM_BEATS[7:0];

    always @(posedge clk) begin
        if (!resetn) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            arr_clr <= 1'b0; arr_en <= 1'b0;
            wbuf_rd_en <= 1'b0; abuf_rd_en <= 1'b0;
            wbuf_rd_addr <= 16'd0; abuf_rd_addr <= 16'd0;
            res_push <= 1'b0; res_tile_i <= 16'd0; res_tile_j <= 16'd0;
            stall_o <= 1'b0;
            k_cnt <= 16'd0; tile_i <= 16'd0; tile_j <= 16'd0; beat <= 8'd0;
            m_q <= 16'd0; n_q <= 16'd0; k_q <= 16'd0;
        end else begin
            // Default-low pulses. Anything that should last exactly one
            // cycle is defaulted here and set explicitly below. This is the
            // same discipline used in dot4_pcpi.v and it prevents the very
            // common "signal stuck high forever" class of bug.
            arr_clr  <= 1'b0;
            arr_en   <= 1'b0;
            res_push <= 1'b0;
            done     <= 1'b0;
            stall_o  <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        // Guard against degenerate dimensions so a
                        // mis-programmed job cannot hang the machine: it
                        // finishes immediately and software sees done with
                        // no results, instead of the accelerator wedging.
                        m_q    <= dim_m;
                        n_q    <= dim_n;
                        k_q    <= dim_k;
                        tile_i <= 16'd0;
                        tile_j <= 16'd0;
                        busy   <= 1'b1;
                        state  <= (dim_m == 16'd0 || dim_n == 16'd0 || dim_k == 16'd0)
                                  ? S_DONE : S_CLEAR;
                    end
                end

                S_CLEAR: begin
                    // Wipe the accumulators before starting a new output
                    // tile. Skipping this is the bug that makes tile 2
                    // contain tile 1's sum added to its own.
                    arr_clr <= 1'b1;
                    k_cnt   <= 16'd0;
                    beat    <= 8'd0;
                    state   <= S_STREAM;
                end

                S_STREAM: begin
                    // Request this step's operands. Addresses are simple
                    // linear walks; a production design would add strides,
                    // but ADDRESSING is not what we are measuring -- the
                    // bandwidth ratio is -- so keeping it simple keeps the
                    // experiment interpretable.
                    wbuf_rd_en   <= 1'b1;
                    wbuf_rd_addr <= k_cnt + tile_j;
                    abuf_rd_en   <= 1'b1;
                    abuf_rd_addr <= k_cnt + tile_i;

                    if (beat_last) begin
                        // Operands are all here: issue one real MAC step.
                        arr_en <= 1'b1;
                        beat   <= 8'd0;
                        if (k_last) state <= S_DRAIN;
                        else        k_cnt <= k_cnt + 16'd1;
                    end else begin
                        // Still waiting on bus beats. THIS is a stall: the
                        // array is ready and willing but has no data. Every
                        // cycle counted here is multiplier hardware sitting
                        // idle -- exactly the waste RQ3 predicts local
                        // buffering removes.
                        beat    <= beat + 8'd1;
                        stall_o <= 1'b1;
                    end
                end

                S_DRAIN: begin
                    wbuf_rd_en <= 1'b0;
                    abuf_rd_en <= 1'b0;
                    res_push   <= 1'b1;
                    res_tile_i <= tile_i;
                    res_tile_j <= tile_j;
                    state      <= S_NEXT;
                end

                S_NEXT: begin
                    // Advance the tile loops: inner over j, outer over i.
                    // This is the hardware form of the two nested for loops
                    // in the header comment.
                    if (!j_last) begin
                        tile_j <= tile_j + ARRAY_W;
                        state  <= S_CLEAR;
                    end else if (!i_last) begin
                        tile_j <= 16'd0;
                        tile_i <= tile_i + ARRAY_H;
                        state  <= S_CLEAR;
                    end else begin
                        state  <= S_DONE;
                    end
                end

                S_DONE: begin
                    // done is a one-cycle pulse (defaulted low above).
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: begin
                    // Unreachable with 6 encodings in 3 bits, but a default
                    // arm costs nothing and guarantees a single upset bit
                    // recovers to a known state instead of wedging forever.
                    state <= S_IDLE;
                end
            endcase
        end
    end

    initial begin
        $display("[accel_ctrl] %0dx%0d WBUF=%0d ABUF=%0d P=%0d -> MEM_BEATS=%0d %s",
                 ARRAY_H, ARRAY_W, WBUF_DEPTH, ABUF_DEPTH, PRECISION, MEM_BEATS,
                 BUFFERED ? "(buffered)" : "(NO BUFFER - control case)");
    end

endmodule
