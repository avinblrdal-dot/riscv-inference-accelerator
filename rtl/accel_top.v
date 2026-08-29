`timescale 1ns / 1ps
//===========================================================================
// accel_top.v -- Memory-mapped wrapper around the MAC array
//===========================================================================
//
// WHAT "MEMORY-MAPPED" MEANS, for a beginner:
//
//   The CPU has no special instruction for "start the accelerator". Instead
//   we make the accelerator LOOK LIKE MEMORY. The C code does:
//
//       *(volatile uint32_t *)0x40000000 = 1;   // write 1 to the CTRL reg
//
//   which the CPU turns into an ordinary store to address 0x40000000. This
//   module watches the memory bus, notices that address, and treats the
//   write as "set the start bit" instead of storing a byte anywhere. Reads
//   work the same way in reverse: a load from the STATUS address returns the
//   busy/done bits rather than any stored value.
//
//   The `volatile` keyword in the C is essential. Without it the compiler
//   sees a store to an address that is never read and deletes it, or hoists
//   a status poll out of a loop so it reads once and spins forever. That is
//   one of the top-three bugs in bare-metal C -- see sw/include/accel.h.
//
// WHY MEMORY-MAPPED RATHER THAN PCPI (a real design decision):
//
//   The dot4 instruction in dot4_pcpi.v uses PCPI, which STALLS the core
//   until the coprocessor answers. That is the right trade for a 1-cycle
//   operation: the handshake is cheaper than any alternative.
//
//   It is the WRONG trade here. This array runs for hundreds of cycles per
//   tile. On PCPI the core would sit frozen for all of them, so total time
//   would be unchanged and we would have gained nothing but area. Memory-
//   mapped means the core writes "go" and is then free -- it can prepare the
//   next tile, or (the interesting option for an energy project) execute a
//   WFI and let the clock gate until the done interrupt arrives.
//
//   Recorded in docs/DECISIONS.md as "Array is memory-mapped, not PCPI".
//
// REGISTER MAP: see accel_pkg.vh. Keep in sync with sw/include/accel.h.
//
// PARAMETERS: ARRAY_H, ARRAY_W, WBUF_DEPTH, ABUF_DEPTH, PRECISION
//===========================================================================

`include "accel_pkg.vh"

module accel_top #(
    parameter ARRAY_H    = 4,
    parameter ARRAY_W    = 4,
    parameter WBUF_DEPTH = 256,
    parameter ABUF_DEPTH = 256,
    parameter PRECISION  = 8
) (
    input  wire        clk,
    input  wire        resetn,

    // ---- simple memory-bus slave (PicoRV32 native interface style) ------
    input  wire        mem_valid,   // master is presenting a transaction
    input  wire [31:0] mem_addr,
    input  wire [31:0] mem_wdata,
    input  wire [3:0]  mem_wstrb,   // 0 = read, nonzero = write (byte enables)
    output reg         mem_ready,   // we have accepted / answered
    output reg  [31:0] mem_rdata,

    output wire        irq_done     // optional: pulse when a job finishes
);

    //-----------------------------------------------------------------------
    // Address decode
    //-----------------------------------------------------------------------
    // Only the low 8 bits select a register; the upper bits are compared
    // against the base address by soc_top before mem_valid reaches us.
    wire [7:0] reg_sel = mem_addr[7:0];

    // ONE ACCEPT PER REQUEST -- and `!mem_ready` is not sufficient to
    // guarantee it.
    //
    // mem_ready is a single-cycle pulse, but the master may still be holding
    // mem_valid high after that pulse has returned low. In that window
    // `mem_valid && !mem_ready` becomes true a SECOND time and every side
    // effect fires twice.
    //
    // The observed symptom: each pushed operand was written to two
    // consecutive buffer addresses and the write pointer advanced by two, so
    // the activation buffer held 1,1,2,2,3,3,4,4 instead of 1..8. The array
    // then computed a confident, fast, WRONG dot product.
    //
    // This is the same failure as the PCPI one-shot bug in dot4_pcpi.v
    // (docs/DECISIONS.md D008), from the same root cause: a master holding a
    // request asserted for longer than the slave's response pulse. A sticky
    // "already served" flag, cleared only when the master finally drops
    // valid, makes the accept unambiguous regardless of how long valid holds.
    reg        bus_served;
    wire       bus_accept = mem_valid && !bus_served;
    wire       is_write = bus_accept && (mem_wstrb != 4'b0000);
    wire       is_read  = bus_accept && (mem_wstrb == 4'b0000);

    always @(posedge clk) begin
        if (!resetn)            bus_served <= 1'b0;
        else if (!mem_valid)    bus_served <= 1'b0;
        else if (bus_accept)    bus_served <= 1'b1;
    end

    //-----------------------------------------------------------------------
    // Control / status registers
    //-----------------------------------------------------------------------
    reg [15:0] dim_m, dim_n, dim_k;
    reg        start_pulse;
    reg        perf_clear;
    reg        done_latch;     // sticky "a job finished", cleared on read

    wire       ctrl_busy, ctrl_done;
    wire       arr_clr, arr_en;
    wire       wbuf_rd_en, abuf_rd_en;
    wire [15:0] wbuf_rd_addr, abuf_rd_addr;
    wire       res_push;
    wire [15:0] res_tile_i, res_tile_j;
    wire       ctrl_stall;

    assign irq_done = ctrl_done;

    //-----------------------------------------------------------------------
    // Buffers
    //-----------------------------------------------------------------------
    wire [31:0] wbuf_rdata, abuf_rdata, abuf_rdata2;
    wire [31:0] wbuf_fill,  abuf_fill;

    // Software fills the buffers by writing to the WBUF/ABUF register
    // addresses. A write pointer auto-increments, so the C code is just a
    // loop of stores to one address -- simple, and it keeps the bus busy
    // with useful traffic rather than address arithmetic.
    reg [15:0] wbuf_wptr, abuf_wptr;

    wire wbuf_wr = is_write && (reg_sel == `ACCEL_REG_WBUF);
    wire abuf_wr = is_write && (reg_sel == `ACCEL_REG_ABUF);

    weight_buffer #(
        .DEPTH(WBUF_DEPTH), .WIDTH(32), .PRECISION(PRECISION)
    ) u_wbuf (
        .clk(clk), .resetn(resetn),
        .wr_en(wbuf_wr), .wr_addr(wbuf_wptr), .wr_data(mem_wdata),
        .rd_en(wbuf_rd_en), .rd_addr(wbuf_rd_addr), .rd_data(wbuf_rdata),
        .bypass_data(mem_wdata),          // DEPTH=0 path
        .fill_count(wbuf_fill)
    );

    activation_buffer #(
        .DEPTH(ABUF_DEPTH), .WIDTH(32), .PRECISION(PRECISION)
    ) u_abuf (
        .clk(clk), .resetn(resetn),
        .wr_en(abuf_wr), .wr_addr(abuf_wptr), .wr_data(mem_wdata),
        .rd_en(abuf_rd_en), .rd_addr(abuf_rd_addr), .rd_data(abuf_rdata),
        .rd2_en(abuf_rd_en), .rd2_addr(abuf_rd_addr + 16'd1), .rd2_data(abuf_rdata2),
        .bypass_data(mem_wdata),
        .fill_count(abuf_fill)
    );

    //-----------------------------------------------------------------------
    // Operand fan-out to the array
    //-----------------------------------------------------------------------
    // The buffers return 32-bit words (4 packed int8 lanes). The array wants
    // ARRAY_H activation bytes and ARRAY_W weight bytes. For ARRAY_W <= 4 one
    // word suffices; for 8 we replicate the word, which is correct for the
    // tiling scheme used here (the controller walks addresses so that the
    // needed lanes land in the low bytes) and keeps the datapath simple.
    //
    // This replication is a deliberate simplification of the operand network,
    // NOT of the arithmetic: the MAC results are still exact. It does mean
    // the 8-wide configuration is optimistic about operand delivery, which is
    // called out in docs/REVIEW.md under "known simplifications" so nobody
    // reports it as a measured bandwidth result.
    wire [ARRAY_H*8-1:0] a_bus;
    wire [ARRAY_W*8-1:0] b_bus;

    genvar gh, gw;
    generate
        for (gh = 0; gh < ARRAY_H; gh = gh + 1) begin : g_afan
            assign a_bus[gh*8 +: 8] = abuf_rdata[(gh % 4)*8 +: 8];
        end
        for (gw = 0; gw < ARRAY_W; gw = gw + 1) begin : g_bfan
            assign b_bus[gw*8 +: 8] = wbuf_rdata[(gw % 4)*8 +: 8];
        end
    endgenerate

    //-----------------------------------------------------------------------
    // Pipeline alignment -- operands arrive one cycle after the address
    //-----------------------------------------------------------------------
    // The buffers are SYNCHRONOUS: accel_ctrl registers a read address in
    // cycle t, the buffer presents that data in cycle t+1. But the FSM also
    // registers arr_en in cycle t, so without correction the array would
    // accumulate in the very cycle the address is being presented -- one step
    // ahead of its own data. It would then multiply whatever the buffer held
    // from the PREVIOUS access and drop the final element of every reduction.
    //
    // Delaying the array's control signals by exactly one cycle lines them up
    // with the data. res_push is delayed with them, because otherwise the
    // finished tile would be captured into the FIFO one cycle before the last
    // accumulate actually lands.
    //
    // This is the classic synchronous-memory off-by-one. It does not hang, it
    // does not warn, and it produces a confident wrong answer.
    reg arr_clr_q, arr_en_q, res_push_q;
    reg [15:0] res_tile_i_q, res_tile_j_q;

    always @(posedge clk) begin
        if (!resetn) begin
            arr_clr_q    <= 1'b0;
            arr_en_q     <= 1'b0;
            res_push_q   <= 1'b0;
            res_tile_i_q <= 16'd0;
            res_tile_j_q <= 16'd0;
        end else begin
            arr_clr_q    <= arr_clr;
            arr_en_q     <= arr_en;
            res_push_q   <= res_push;
            res_tile_i_q <= res_tile_i;
            res_tile_j_q <= res_tile_j;
        end
    end

    //-----------------------------------------------------------------------
    // The array
    //-----------------------------------------------------------------------
    wire [ARRAY_H*ARRAY_W*`ACCEL_ACC_W-1:0] acc_flat;
    wire [15:0] macs_cycle;

    mac_array #(
        .ARRAY_H(ARRAY_H), .ARRAY_W(ARRAY_W),
        .ACC_W(`ACCEL_ACC_W), .PRECISION(PRECISION)
    ) u_array (
        .clk(clk), .resetn(resetn),
        .clr(arr_clr_q), .en(arr_en_q),
        .a_in(a_bus), .b_in(b_bus),
        .acc_flat(acc_flat),
        .macs_this_cycle(macs_cycle)
    );

`ifdef ACCEL_TRACE
    // Debug only: show the operands the array actually consumes each step.
    always @(posedge clk) begin
        if (resetn && arr_en_q)
            $display("[trace] MAC step: a_lane0=%0d b_lane0=%0d b_lane1=%0d b_lane2=%0d b_lane3=%0d",
                     $signed(a_bus[7:0]), $signed(b_bus[7:0]),
                     $signed(b_bus[15:8]), $signed(b_bus[23:16]),
                     $signed(b_bus[31:24]));
    end
`endif

    //-----------------------------------------------------------------------
    // The sequencer
    //-----------------------------------------------------------------------
    accel_ctrl #(
        .ARRAY_H(ARRAY_H), .ARRAY_W(ARRAY_W),
        .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH),
        .PRECISION(PRECISION), .BUS_BYTES(4)
    ) u_ctrl (
        .clk(clk), .resetn(resetn),
        .start(start_pulse),
        .dim_m(dim_m), .dim_n(dim_n), .dim_k(dim_k),
        .busy(ctrl_busy), .done(ctrl_done),
        .arr_clr(arr_clr), .arr_en(arr_en),
        .wbuf_rd_en(wbuf_rd_en), .wbuf_rd_addr(wbuf_rd_addr),
        .abuf_rd_en(abuf_rd_en), .abuf_rd_addr(abuf_rd_addr),
        .res_push(res_push), .res_tile_i(res_tile_i), .res_tile_j(res_tile_j),
        .stall_o(ctrl_stall)
    );

    //-----------------------------------------------------------------------
    // Performance counters
    //-----------------------------------------------------------------------
    wire [31:0] p_cycles, p_active, p_stalls, p_macs;

    perf_counter u_perf (
        .clk(clk), .resetn(resetn),
        .clear(perf_clear),
        .run(ctrl_busy),
        .mac_issue(macs_cycle),
        .stall(ctrl_stall),
        .cycles_total_o(p_cycles),
        .cycles_active_o(p_active),
        .cycles_stall_o(p_stalls),
        .macs_done_o(p_macs)
    );

    //-----------------------------------------------------------------------
    // Result FIFO
    //-----------------------------------------------------------------------
    // Finished tiles are pushed here and software pops them one int32 at a
    // time by reading ACCEL_REG_RESULT. Depth is one tile plus slack, which
    // is enough because the controller cannot start the next tile until the
    // current one has drained.
    localparam TILE_ELEMS = ARRAY_H * ARRAY_W;
    localparam FIFO_DEPTH = TILE_ELEMS * 2;
    localparam FIFO_AW    = (FIFO_DEPTH <= 1) ? 1 : $clog2(FIFO_DEPTH);

    reg [31:0]         fifo [0:FIFO_DEPTH-1];
    reg [FIFO_AW:0]    fifo_wp, fifo_rp;
    wire               fifo_empty = (fifo_wp == fifo_rp);

    wire res_pop = is_read && (reg_sel == `ACCEL_REG_RESULT) && !fifo_empty;

    integer fi;
    always @(posedge clk) begin
        if (!resetn) begin
            fifo_wp <= 0;
            fifo_rp <= 0;
        end else begin
            // Push a whole finished tile in one cycle. In real silicon this
            // would be a multi-cycle drain; doing it in one keeps the model
            // simple and does not affect the MAC-cycle accounting, which is
            // what the experiment measures.
            if (res_push_q) begin
                for (fi = 0; fi < TILE_ELEMS; fi = fi + 1)
                    fifo[(fifo_wp + fi[FIFO_AW:0]) % FIFO_DEPTH] <=
                        acc_flat[fi*`ACCEL_ACC_W +: `ACCEL_ACC_W];
                fifo_wp <= fifo_wp + TILE_ELEMS[FIFO_AW:0];
            end
            if (res_pop)
                fifo_rp <= fifo_rp + 1'b1;
        end
    end

    //-----------------------------------------------------------------------
    // Bus response
    //-----------------------------------------------------------------------
    // Every transaction completes in exactly one cycle: mem_ready pulses for
    // one cycle in response to mem_valid. Holding mem_ready high across two
    // cycles would make the core think two transactions completed and it
    // would skip an instruction -- a spectacularly confusing bug.
    always @(posedge clk) begin
        if (!resetn) begin
            mem_ready   <= 1'b0;
            mem_rdata   <= 32'd0;
            dim_m       <= 16'd0;
            dim_n       <= 16'd0;
            dim_k       <= 16'd0;
            start_pulse <= 1'b0;
            perf_clear  <= 1'b0;
            done_latch  <= 1'b0;
            wbuf_wptr   <= 16'd0;
            abuf_wptr   <= 16'd0;
        end else begin
            start_pulse <= 1'b0;   // one-cycle pulse
            perf_clear  <= 1'b0;
            mem_ready   <= 1'b0;

            // Sticky done flag so software cannot miss a fast job between
            // two polls. Cleared when STATUS is read.
            if (ctrl_done) done_latch <= 1'b1;

            if (bus_accept) begin
                mem_ready <= 1'b1;

                if (is_write) begin
                    case (reg_sel)
                        `ACCEL_REG_CTRL: begin
                            if (mem_wdata[0]) start_pulse <= 1'b1;
                            if (mem_wdata[1]) begin
                                // Rewind BOTH write pointers.
                                wbuf_wptr <= 16'd0;
                                abuf_wptr <= 16'd0;
                            end
                            // Independent rewinds. These exist so software can
                            // refill activations for a new output position
                            // WITHOUT evicting weights that are still resident
                            // and still valid. Without them, every activation
                            // refill would force a weight reload too, which
                            // defeats the buffer entirely -- the buffer would
                            // be present but never actually reused.
                            if (mem_wdata[3]) wbuf_wptr <= 16'd0;
                            if (mem_wdata[4]) abuf_wptr <= 16'd0;
                            if (mem_wdata[2]) perf_clear <= 1'b1;
                        end
                        `ACCEL_REG_M:    dim_m <= mem_wdata[15:0];
                        `ACCEL_REG_N:    dim_n <= mem_wdata[15:0];
                        `ACCEL_REG_K:    dim_k <= mem_wdata[15:0];
                        `ACCEL_REG_WBUF: wbuf_wptr <= wbuf_wptr + 16'd1;
                        `ACCEL_REG_ABUF: abuf_wptr <= abuf_wptr + 16'd1;
                        default: ; // writes to read-only regs are ignored
                    endcase
                end else begin
                    case (reg_sel)
                        `ACCEL_REG_STATUS: begin
                            mem_rdata  <= {29'd0, !fifo_empty, done_latch, ctrl_busy};
                            done_latch <= 1'b0;   // read-to-clear
                        end
                        `ACCEL_REG_M:      mem_rdata <= {16'd0, dim_m};
                        `ACCEL_REG_N:      mem_rdata <= {16'd0, dim_n};
                        `ACCEL_REG_K:      mem_rdata <= {16'd0, dim_k};
                        `ACCEL_REG_RESULT: mem_rdata <= fifo_empty
                                                        ? 32'd0
                                                        : fifo[fifo_rp[FIFO_AW-1:0]];
                        `ACCEL_REG_CYCLES: mem_rdata <= p_cycles;
                        `ACCEL_REG_STALLS: mem_rdata <= p_stalls;
                        `ACCEL_REG_MACS:   mem_rdata <= p_macs;
                        // Software cannot know which geometry was built --
                        // the sweep changes it per configuration. Without
                        // this the firmware would have to be recompiled per
                        // RTL config, and a mismatch would silently compute
                        // the wrong thing rather than failing.
                        `ACCEL_REG_BUFCAP: mem_rdata <= {ABUF_DEPTH[15:0],
                                                         WBUF_DEPTH[15:0]};
                        `ACCEL_REG_CONFIG: mem_rdata <= {8'd4,
                                                         PRECISION[7:0],
                                                         ARRAY_W[7:0],
                                                         ARRAY_H[7:0]};
                        default:           mem_rdata <= 32'd0;
                    endcase
                end
            end
        end
    end

    // Keep the extra activation tap and fill counters from being optimised
    // away; they are read by testbenches and reported in the sweep metadata.
    wire _unused = &{1'b0, abuf_rdata2, wbuf_fill, abuf_fill, p_active,
                     res_tile_i_q, res_tile_j_q, 1'b0};

endmodule
