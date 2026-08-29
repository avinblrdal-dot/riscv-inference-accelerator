`timescale 1ns / 1ps
//===========================================================================
// soc_top.v -- The whole system: CPU + memory + UART + accelerator
//===========================================================================
//
// "SoC" = System on Chip. This module is the thing that actually gets
// synthesised onto the FPGA, and the thing sim/tb_soc.v simulates. It wires
// together four pieces:
//
//     picorv32   the RISC-V CPU core (third_party, we did not write it)
//     memory     one block of RAM holding both the program and its data
//     uart_tx    so printf() has somewhere to go
//     accel_top  our MAC array, memory-mapped
//     dot4_pcpi  our custom instructions, on the coprocessor port
//
// THE ADDRESS MAP -- how one bus serves four devices:
//
//   The CPU emits an address and expects somebody to answer. This module is
//   the traffic warden: it looks at the top bits of the address and routes
//   the transaction to exactly one device, then routes that device's answer
//   back. Everything is one cycle.
//
//     0x0000_0000 .. MEM_BYTES-1   RAM (program + data + stack)
//     0x1000_0000                  UART data register (write a byte to send)
//     0x1000_0004                  UART status  (bit0 = busy)
//     0x2000_0000                  cycle counter (read)
//     0x2000_0004                  instruction-retired counter (read)
//     0x3000_0000                  simulation exit  (write -> $finish)
//     0x4000_0000 .. +0xFF         accelerator registers (see accel_pkg.vh)
//
//   THE #1 BEGINNER BUG HERE: if no device claims an address, mem_ready is
//   never asserted and the CPU waits forever. The simulation does not crash
//   -- it just goes silent. The "unmapped address" catch-all at the bottom
//   of this file answers such accesses and prints a loud warning, which
//   turns a mystery hang into a one-line diagnosis. Do not delete it.
//
// PARAMETERS: the sweep drives ARRAY_H/ARRAY_W/WBUF_DEPTH/ABUF_DEPTH/
//             PRECISION straight through to the accelerator, and
//             ENABLE_ACCEL/ENABLE_DOT4 select which variant is built so the
//             baseline can be measured with the extra hardware genuinely
//             absent rather than merely unused.
//===========================================================================

`include "accel_pkg.vh"

module soc_top #(
    parameter MEM_WORDS     = 16384,          // 64 KB of RAM
    parameter CLKS_PER_BIT  = 868,            // 100 MHz / 115200 baud
    parameter ENABLE_DOT4   = 1,              // build the PCPI coprocessor?
    parameter ENABLE_ACCEL  = 1,              // build the MAC array?
    parameter ARRAY_H       = 4,
    parameter ARRAY_W       = 4,
    parameter WBUF_DEPTH    = 256,
    parameter ABUF_DEPTH    = 256,
    parameter PRECISION     = 8,
    parameter FIRMWARE_HEX  = "firmware.hex"  // loaded by $readmemh in sim
) (
    input  wire clk,
    input  wire resetn,
    output wire uart_tx_pin,
    output wire trap                          // CPU hit an illegal instruction
);

    localparam MEM_BYTES = MEM_WORDS * 4;

    //-----------------------------------------------------------------------
    // CPU <-> bus signals
    //-----------------------------------------------------------------------
    wire        mem_valid, mem_instr;
    reg         mem_ready;
    wire [31:0] mem_addr, mem_wdata;
    wire [3:0]  mem_wstrb;
    reg  [31:0] mem_rdata;

    wire        pcpi_valid;
    wire [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2;
    wire        pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_rd;

    //-----------------------------------------------------------------------
    // The CPU
    //-----------------------------------------------------------------------
    // ENABLE_COUNTERS gives us rdcycle/rdinstret, which sw/include/perf.h
    // reads to time code regions from inside C -- that is how the baseline
    // MAC-loop fraction for RQ1 gets measured.
    //
    // BARREL_SHIFTER and TWO_STAGE_SHIFT are left at defaults deliberately:
    // changing them changes the baseline's instruction timing, and the
    // baseline is the thing every speedup is measured against. If you ever
    // change a core parameter, every previously collected number is invalid
    // and the sweep must be re-run. Record it in docs/DECISIONS.md.
    picorv32 #(
        .ENABLE_COUNTERS   (1),
        .ENABLE_COUNTERS64 (0),   // 32-bit counters are enough and cheaper
        .ENABLE_MUL        (0),   // no hardware multiply: the whole point is
                                  // that plain rv32i does MACs slowly
        .ENABLE_DIV        (0),
        .ENABLE_IRQ        (0),
        .ENABLE_PCPI       (ENABLE_DOT4),
        .COMPRESSED_ISA    (0),
        .PROGADDR_RESET    (32'h0000_0000),
        .STACKADDR         (MEM_BYTES)
    ) u_cpu (
        .clk        (clk),
        .resetn     (resetn),
        .trap       (trap),
        .mem_valid  (mem_valid),
        .mem_instr  (mem_instr),
        .mem_ready  (mem_ready),
        .mem_addr   (mem_addr),
        .mem_wdata  (mem_wdata),
        .mem_wstrb  (mem_wstrb),
        .mem_rdata  (mem_rdata),
        .mem_la_read(), .mem_la_write(), .mem_la_addr(),
        .mem_la_wdata(), .mem_la_wstrb(),
        .pcpi_valid (pcpi_valid),
        .pcpi_insn  (pcpi_insn),
        .pcpi_rs1   (pcpi_rs1),
        .pcpi_rs2   (pcpi_rs2),
        .pcpi_wr    (pcpi_wr),
        .pcpi_rd    (pcpi_rd),
        .pcpi_wait  (pcpi_wait),
        .pcpi_ready (pcpi_ready),
        .irq        (32'd0),
        .eoi        (),
        // ENABLE_TRACE is 0, so these outputs are unused. Listing them
        // explicitly (rather than omitting them) documents that the omission
        // is deliberate and silences Verilator's PINMISSING warning, which
        // otherwise hides real missing connections in the noise.
        .trace_valid(),
        .trace_data ()
    );

    //-----------------------------------------------------------------------
    // The custom-instruction coprocessor
    //-----------------------------------------------------------------------
    generate
        if (ENABLE_DOT4) begin : g_dot4
            dot4_pcpi #(
                .PIPELINE  (0),
                .PRECISION (PRECISION)
            ) u_dot4 (
                .clk(clk), .resetn(resetn),
                .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
                .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
                .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
                .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
                .acc_value(), .acc_active()
            );
        end else begin : g_nodot4
            // Baseline build: the coprocessor is genuinely ABSENT, not just
            // unused, so its area does not appear in the synthesis report.
            // Tying ready/wait low means any custom-0 instruction traps as
            // illegal -- which is what we want, because it turns "you forgot
            // to enable the accelerator" into an immediate visible trap
            // rather than a silently wrong answer.
            assign pcpi_wr    = 1'b0;
            assign pcpi_rd    = 32'd0;
            assign pcpi_wait  = 1'b0;
            assign pcpi_ready = 1'b0;
        end
    endgenerate

    //-----------------------------------------------------------------------
    // Address decode
    //-----------------------------------------------------------------------
    wire sel_ram   = mem_valid && (mem_addr <  MEM_BYTES);
    wire sel_uart  = mem_valid && (mem_addr[31:28] == 4'h1);
    wire sel_perf  = mem_valid && (mem_addr[31:28] == 4'h2);
    wire sel_exit  = mem_valid && (mem_addr[31:28] == 4'h3);
    wire sel_accel = mem_valid && (mem_addr[31:28] == 4'h4);

    //-----------------------------------------------------------------------
    // RAM
    //-----------------------------------------------------------------------
    // One memory holding instructions AND data (a "von Neumann" layout).
    // Loaded at time 0 from a hex file produced by the RISC-V toolchain.
    reg [31:0] ram [0:MEM_WORDS-1];
    reg [31:0] ram_rdata;

    initial begin
        // $readmemh fills the array from a text file of hex words. If the
        // file is missing, Icarus warns and leaves memory as X (unknown),
        // and the CPU immediately executes garbage. If your simulation dies
        // at time ~0 with a trap, CHECK THIS FILE EXISTS first.
        if (FIRMWARE_HEX != "")
            $readmemh(FIRMWARE_HEX, ram);
    end

    wire [$clog2(MEM_WORDS)-1:0] ram_word_addr = mem_addr[$clog2(MEM_WORDS)+1:2];

    always @(posedge clk) begin
        if (sel_ram) begin
            // Byte-enables: the CPU can write 1, 2 or 4 bytes. Ignoring
            // mem_wstrb and always writing 32 bits corrupts neighbouring
            // bytes -- the classic symptom is strings printing with garbage
            // in the middle, because a char store clobbered its neighbours.
            if (mem_wstrb[0]) ram[ram_word_addr][ 7: 0] <= mem_wdata[ 7: 0];
            if (mem_wstrb[1]) ram[ram_word_addr][15: 8] <= mem_wdata[15: 8];
            if (mem_wstrb[2]) ram[ram_word_addr][23:16] <= mem_wdata[23:16];
            if (mem_wstrb[3]) ram[ram_word_addr][31:24] <= mem_wdata[31:24];
            ram_rdata <= ram[ram_word_addr];
        end
    end

    //-----------------------------------------------------------------------
    // UART
    //-----------------------------------------------------------------------
    wire       uart_busy;
    wire       uart_start = sel_uart && (mem_wstrb != 4'b0) && (mem_addr[3:0] == 4'h0);

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_uart (
        .clk(clk), .resetn(resetn),
        .tx_start(uart_start && !uart_busy),
        .tx_data (mem_wdata[7:0]),
        .tx      (uart_tx_pin),
        .tx_busy (uart_busy)
    );

    //-----------------------------------------------------------------------
    // Free-running system counters
    //-----------------------------------------------------------------------
    // These count EVERYTHING, including cycles the CPU spends waiting on the
    // accelerator. That is deliberate: wall-clock cycles are what determines
    // energy per inference, and a measurement that excluded stall time would
    // flatter the accelerator.
    reg [31:0] cycle_count, instr_count;
    always @(posedge clk) begin
        if (!resetn) begin
            cycle_count <= 32'd0;
            instr_count <= 32'd0;
        end else begin
            cycle_count <= cycle_count + 32'd1;
            // Count instruction fetches that completed.
            if (mem_valid && mem_ready && mem_instr)
                instr_count <= instr_count + 32'd1;
        end
    end

    //-----------------------------------------------------------------------
    // Accelerator
    //-----------------------------------------------------------------------
    wire        accel_ready;
    wire [31:0] accel_rdata;

    generate
        if (ENABLE_ACCEL) begin : g_accel
            accel_top #(
                .ARRAY_H(ARRAY_H), .ARRAY_W(ARRAY_W),
                .WBUF_DEPTH(WBUF_DEPTH), .ABUF_DEPTH(ABUF_DEPTH),
                .PRECISION(PRECISION)
            ) u_accel (
                .clk(clk), .resetn(resetn),
                .mem_valid(sel_accel),
                .mem_addr (mem_addr),
                .mem_wdata(mem_wdata),
                .mem_wstrb(mem_wstrb),
                .mem_ready(accel_ready),
                .mem_rdata(accel_rdata),
                .irq_done ()
            );
        end else begin : g_noaccel
            // Baseline build. Still ANSWER the bus so that software which
            // accidentally pokes the accelerator hangs nothing -- it just
            // reads zero. A hang here would be indistinguishable from a
            // firmware bug.
            assign accel_ready = sel_accel;
            assign accel_rdata = 32'd0;
        end
    endgenerate

    //-----------------------------------------------------------------------
    // Bus response multiplexer
    //-----------------------------------------------------------------------
    // Exactly one device answers each transaction, and mem_ready is a
    // single-cycle pulse. The `!mem_ready` term is what makes it a pulse:
    // without it, ready would stay high for as long as valid is high and the
    // CPU would believe several transactions had completed.
    always @(posedge clk) begin
        if (!resetn) begin
            mem_ready <= 1'b0;
            mem_rdata <= 32'd0;
        end else begin
            mem_ready <= 1'b0;

            if (mem_valid && !mem_ready) begin
                if (sel_ram) begin
                    mem_ready <= 1'b1;
                    mem_rdata <= ram[ram_word_addr];
                end else if (sel_uart) begin
                    mem_ready <= 1'b1;
                    // Reading the status register lets software wait for the
                    // transmitter instead of guessing a delay.
                    mem_rdata <= (mem_addr[3:0] == 4'h4) ? {31'd0, uart_busy}
                                                         : 32'd0;
                end else if (sel_perf) begin
                    mem_ready <= 1'b1;
                    mem_rdata <= (mem_addr[3:0] == 4'h0) ? cycle_count
                               : (mem_addr[3:0] == 4'h4) ? instr_count
                                                         : 32'd0;
                end else if (sel_exit) begin
`ifndef SYNTHESIS
                    // Simulation-only: a write here ends the run cleanly and
                    // carries an exit code, so run_icarus.sh can tell a
                    // passing firmware from a failing one. Guarded out of
                    // synthesis because $display/$finish are not hardware.
                    $display("[soc] firmware requested exit, code=%0d", mem_wdata);
                    $finish;
`endif
                    mem_ready <= 1'b1;
                end else if (sel_accel) begin
                    mem_ready <= accel_ready;
                    mem_rdata <= accel_rdata;
                end else begin
                    //-----------------------------------------------------
                    // UNMAPPED ADDRESS -- the anti-hang guard.
                    //-----------------------------------------------------
                    // Answer anyway (so the CPU proceeds) but shout about
                    // it. Silence here is the single most confusing failure
                    // mode in a small SoC.
                    mem_ready <= 1'b1;
                    mem_rdata <= 32'd0;
`ifndef SYNTHESIS
                    $display("[soc] WARNING unmapped %s at 0x%08x (t=%0t) -- check your address map",
                             (mem_wstrb != 0) ? "WRITE" : "READ", mem_addr, $time);
`endif
                end
            end
        end
    end

    wire _unused = &{1'b0, ram_rdata, 1'b0};

endmodule
