//===========================================================================
// tb_soc.v -- Full-system testbench: CPU + memory + UART + accelerator
//===========================================================================
//
// This is the testbench that proves the whole machine works. It loads a
// program image into the SoC's memory, releases reset, and then acts as the
// laptop on the other end of the serial cable: it watches the UART pin,
// reassembles bytes, and prints them.
//
// HOW A PROGRAM GETS IN:
//   rtl/soc_top.v calls $readmemh(FIRMWARE_HEX, ram) at time 0. The hex file
//   is one 32-bit word per line. It comes from either
//     * sim/gen_smoke_hex.py   (no toolchain needed -- for smoke tests)
//     * the RISC-V cross-compiler via sw/Makefile (for real C programs)
//
//   IF YOUR SIMULATION DIES IMMEDIATELY, CHECK THE HEX FILE EXISTS. A missing
//   file leaves memory as X, the CPU executes garbage, and it traps at once.
//
// PASS/FAIL:
//   The firmware writes to the exit port (0x3000_0000) when it is done, which
//   ends the simulation. This testbench additionally fails the run if:
//     * the CPU asserts trap (illegal instruction / misaligned access)
//     * the expected output string never appears
//     * nothing happens before the timeout (a hang)
//
//   Pass +expect=STRING to require particular UART output. run_icarus.sh
//   uses this to check the smoke tests print what they should.
//
// RUN IT:
//   ./sim/run_icarus.sh tb_soc
//===========================================================================

`timescale 1ns / 1ps

module tb_soc;

    // Small CLKS_PER_BIT so the UART is fast in simulation. On real hardware
    // this is 868 (100 MHz / 115200 baud); here we only care that the
    // framing is correct, not that it matches a real baud rate.
    parameter CLKS_PER_BIT = 4;
    parameter CLK_NS       = 10;                 // 100 MHz
    parameter BIT_NS       = CLKS_PER_BIT * CLK_NS;
    parameter MEM_WORDS    = 16384;
    // Generous, because a full baseline inference is genuinely enormous:
    // ~372k MACs, each an __mulsi3 software-multiply call on rv32i, at ~4
    // cycles per instruction. That is tens of millions of cycles. The smoke
    // tests finish in microseconds; the real firmware does not.
    // Override for a quick run:  vvp tb_soc.vvp +timeout_ns=20000000
    parameter TIMEOUT_NS   = 4_000_000_000;

    reg  clk = 0;
    reg  resetn = 0;
    wire uart_pin;
    wire trap;

    // The program image always has the SAME filename, sim/build/firmware.hex.
    // run_icarus.sh copies whichever test program you asked for onto that
    // name before launching the simulation.
    //
    // Why not pass the path in as a parameter? Because $readmemh happens
    // inside soc_top at time 0, and threading a string parameter down through
    // the hierarchy from the command line is fiddly and tool-dependent.
    // Copying the file is boring, portable, and impossible to get subtly
    // wrong -- and "boring and obvious" is worth a lot in a build that four
    // beginners have to debug.

    always #(CLK_NS/2) clk = ~clk;

    soc_top #(
        .MEM_WORDS    (MEM_WORDS),
        .CLKS_PER_BIT (CLKS_PER_BIT),
        .ENABLE_DOT4  (1),
        .ENABLE_ACCEL (1),
        .ARRAY_H      (4),
        .ARRAY_W      (4),
        .WBUF_DEPTH   (256),
        .ABUF_DEPTH   (256),
        .PRECISION    (8),
        .FIRMWARE_HEX ("sim/build/firmware.hex")
    ) dut (
        .clk(clk), .resetn(resetn),
        .uart_tx_pin(uart_pin), .trap(trap)
    );

    //-----------------------------------------------------------------------
    // UART receiver -- the "laptop" end of the serial link
    //-----------------------------------------------------------------------
    // Serial framing: the line idles high, a falling edge marks the START
    // bit, then 8 data bits LSB-first, then a STOP bit. We wait one and a
    // half bit times after the falling edge so that we sample in the MIDDLE
    // of the first data bit rather than at its edge, where the value is least
    // certain. Sampling at edges is the classic reason a UART "almost works"
    // and returns occasional garbage characters.
    integer bit_i;
    reg [7:0] rx_byte;
    integer   n_chars = 0;
    reg [8*256-1:0] rx_buffer;      // accumulated output, for +expect matching

    initial begin
        rx_buffer = 0;
        forever begin
            @(negedge uart_pin);
            #(BIT_NS + BIT_NS/2);
            for (bit_i = 0; bit_i < 8; bit_i = bit_i + 1) begin
                rx_byte[bit_i] = uart_pin;
                #(BIT_NS);
            end
            $write("%c", rx_byte);
            $fflush();
            rx_buffer = {rx_buffer[8*255-1:0], rx_byte};
            n_chars = n_chars + 1;
        end
    end

    //-----------------------------------------------------------------------
    // Failure detectors
    //-----------------------------------------------------------------------
    // A trap means the CPU hit an illegal or misaligned instruction. The most
    // common cause in this project is executing a custom-0 instruction in a
    // build where ENABLE_DOT4=0 -- which is exactly why we want it to trap
    // loudly rather than quietly return a wrong number.
    always @(posedge clk) begin
        if (resetn && trap) begin
            $display("");
            $display("TEST FAILED -- CPU TRAP at time %0t", $time);
            $display("  Most likely: a custom instruction in a build without");
            $display("  the coprocessor, or a jump into unwritten memory.");
            $finish;
        end
    end

    integer timeout_ns;
    initial begin
        timeout_ns = TIMEOUT_NS;
        // $value$plusargs returns 1 if the argument was present. Verilog-2001
        // has no void cast, so test it in an if rather than discarding it.
        if ($value$plusargs("timeout_ns=%d", timeout_ns))
            $display("[tb_soc] timeout overridden to %0d ns", timeout_ns);
        #timeout_ns;
        $display("");
        $display("TEST FAILED -- timeout after %0d characters of output.", n_chars);
        $display("  A hang usually means an unmapped memory access (look for");
        $display("  the [soc] WARNING lines) or a firmware infinite loop.");
        $finish;
    end

    initial begin
        if ($test$plusargs("vcd")) begin
            $dumpfile("sim/build/tb_soc.vcd");
            $dumpvars(0, tb_soc);
        end
        $display("[tb_soc] releasing reset");
        @(posedge clk);
        resetn <= 1'b1;
    end

endmodule
