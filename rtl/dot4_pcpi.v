`timescale 1ns / 1ps
//===========================================================================
// dot4_pcpi.v -- Custom RISC-V instructions on PicoRV32's coprocessor port
//===========================================================================
//
// THE CONCEPT, for a beginner:
//
//   A CPU has a fixed instruction set. PicoRV32 implements rv32i -- add,
//   load, store, branch, and so on. When it fetches a 32-bit word it does
//   not recognise, it does not immediately crash. Instead, if built with
//   ENABLE_PCPI=1, it offers the instruction to an external module over the
//   "Pico Co-Processor Interface" (PCPI) and waits to see if anybody claims
//   it. That external module is this file.
//
//   So we are literally adding new instructions to the processor. Software
//   written in C can then call them (see sw/include/accel.h) and they run in
//   a couple of cycles instead of the ~20 instructions the same arithmetic
//   would take in plain rv32i.
//
// THE HANDSHAKE (get this wrong and the whole CPU freezes):
//
//   core -> us :  pcpi_valid    "here is an instruction I don't know"
//                 pcpi_insn     the full 32-bit instruction word
//                 pcpi_rs1/rs2  the VALUES already read from the registers
//   us -> core :  pcpi_wr       "I am writing a result to rd"
//                 pcpi_rd       the result value
//                 pcpi_ready    "I am done, take it"
//                 pcpi_wait     "this is mine, but I need more cycles"
//
//   The rule that matters: if we assert pcpi_wait and NEVER assert
//   pcpi_ready, the core stalls forever. That is a hang with no error
//   message -- the simulation just goes quiet. sim/tb_dot4.v has an explicit
//   test for this (see "deadlock guard"), because it is the single easiest
//   way to lose a weekend on this project.
//
//   Equally important: if the instruction is NOT ours, we must assert
//   NEITHER wait NOR ready, so the core can trap it as illegal.
//
// THE INSTRUCTIONS:
//
//   DOT4  rd, rs1, rs2   rd = sum over 4 lanes of int8(rs1[i]) * int8(rs2[i])
//   DOT4A rd, rs1, rs2   acc += that sum   (rd is not written)
//   ACCRD rd, x0,  x0    rd = acc; acc = 0  (read-and-clear)
//
//   Packing: rs1[7:0] is lane 0, rs1[15:8] lane 1, and so on. Each lane is a
//   SIGNED 8-bit value. The signedness is the classic bug in this module --
//   see the $signed() casts below and read the comment there.
//
// PARAMETERS:
//   PIPELINE  0 = combinational multiply, answer ready the same cycle (1
//                 cycle latency). Simplest, but the 4 multiplies plus adder
//                 tree sit in one clock period and may fail timing closure
//                 on the FPGA at higher clock rates.
//             1 = register the products, answer one cycle later (2 cycle
//                 latency). Shorter critical path, higher achievable Fmax.
//             This is the "escape hatch" for timing closure -- flip it to 1
//             and re-run synthesis before redesigning anything.
//
// PORTS: see the declarations; all are PicoRV32-defined except clk/resetn.
//===========================================================================

`include "accel_pkg.vh"

module dot4_pcpi #(
    parameter PIPELINE  = 0,   // 0 = 1-cycle (comb mult), 1 = 2-cycle
    parameter PRECISION = 8    // 8 or 4; 4 sign-extends nibbles (see below)
) (
    input  wire        clk,
    input  wire        resetn,      // ACTIVE LOW reset (PicoRV32 convention)

    // ---- PCPI slave interface -------------------------------------------
    input  wire        pcpi_valid,
    input  wire [31:0] pcpi_insn,
    input  wire [31:0] pcpi_rs1,
    input  wire [31:0] pcpi_rs2,
    output reg         pcpi_wr,
    output reg  [31:0] pcpi_rd,
    output reg         pcpi_wait,
    output reg         pcpi_ready,

    // ---- Observability (for testbenches and perf counters) --------------
    output wire [31:0] acc_value,   // current accumulator, for debug
    output wire        acc_active   // high while this unit owns an insn
);

    //-----------------------------------------------------------------------
    // 1. Decode -- is this instruction ours?
    //-----------------------------------------------------------------------
    // Pure combinational wires. In Verilog, "wire" with assign is a
    // continuous connection: whenever the right side changes, the left side
    // changes. There is no "when" -- it is a permanent equation, like a
    // spreadsheet formula, not a Python statement that runs once.
    wire [6:0] opcode = pcpi_insn[6:0];
    wire [2:0] funct3 = pcpi_insn[14:12];
    wire [6:0] funct7 = pcpi_insn[31:25];

    wire is_custom0 = (opcode == `ACCEL_OPCODE_CUSTOM0) &&
                      (funct3 == `ACCEL_FUNCT3_DOT);

    wire is_dot4    = is_custom0 && (funct7 == `ACCEL_FUNCT7_DOT4);
    wire is_dot4a   = is_custom0 && (funct7 == `ACCEL_FUNCT7_DOT4A);
    wire is_accrd   = is_custom0 && (funct7 == `ACCEL_FUNCT7_ACCRD);

    // "Claimed" means: the core is offering an instruction AND it is one of
    // ours. If claimed is low we keep every output deasserted so the core
    // can raise an illegal-instruction trap instead of hanging.
    wire claimed    = pcpi_valid && (is_dot4 || is_dot4a || is_accrd);

    assign acc_active = claimed;

    //-----------------------------------------------------------------------
    // 2. Unpack the four signed lanes
    //-----------------------------------------------------------------------
    // THE CLASSIC BUG LIVES HERE. In Verilog, a bit-select like pcpi_rs1[7:0]
    // is UNSIGNED by default. If lane 0 holds 0xFF and you multiply it
    // without a cast, you get 255 * x, not -1 * x. Every downstream number
    // is then wrong, and because the error is data-dependent it only shows
    // up on some test vectors -- which is the worst kind of bug.
    //
    // $signed() reinterprets the bits as two's complement, so 0xFF becomes
    // -1. The products are then computed in signed arithmetic.
    //
    // Verilog's rule: an expression is signed only if EVERY operand is
    // signed. One stray unsigned operand silently makes the whole expression
    // unsigned. That is why both sides of each multiply are cast.
    //
    // PRECISION=4: the value lives in the low nibble of each byte. We
    // sign-extend the nibble to 8 bits so the rest of the datapath is
    // identical. Synthesis then sees a narrower multiply and infers less
    // logic -- which is exactly the area saving the sweep is trying to
    // measure, so this must not be optimised away by accident.
    wire signed [7:0] a0, a1, a2, a3;
    wire signed [7:0] b0, b1, b2, b3;

    generate
        if (PRECISION == 4) begin : g_nibble
            // {{4{x[3]}}, x[3:0]} replicates the sign bit 4 times, then
            // appends the nibble: the standard sign-extension idiom.
            assign a0 = {{4{pcpi_rs1[ 3]}}, pcpi_rs1[ 3: 0]};
            assign a1 = {{4{pcpi_rs1[11]}}, pcpi_rs1[11: 8]};
            assign a2 = {{4{pcpi_rs1[19]}}, pcpi_rs1[19:16]};
            assign a3 = {{4{pcpi_rs1[27]}}, pcpi_rs1[27:24]};
            assign b0 = {{4{pcpi_rs2[ 3]}}, pcpi_rs2[ 3: 0]};
            assign b1 = {{4{pcpi_rs2[11]}}, pcpi_rs2[11: 8]};
            assign b2 = {{4{pcpi_rs2[19]}}, pcpi_rs2[19:16]};
            assign b3 = {{4{pcpi_rs2[27]}}, pcpi_rs2[27:24]};
        end else begin : g_byte
            assign a0 = pcpi_rs1[ 7: 0];
            assign a1 = pcpi_rs1[15: 8];
            assign a2 = pcpi_rs1[23:16];
            assign a3 = pcpi_rs1[31:24];
            assign b0 = pcpi_rs2[ 7: 0];
            assign b1 = pcpi_rs2[15: 8];
            assign b2 = pcpi_rs2[23:16];
            assign b3 = pcpi_rs2[31:24];
        end
    endgenerate

    //-----------------------------------------------------------------------
    // 3. The arithmetic
    //-----------------------------------------------------------------------
    // Four 8x8 signed multiplies -> 16-bit products -> sign-extended to 32
    // and summed. Sign extension before the adds matters: adding 16-bit
    // signed numbers in a 32-bit context requires the compiler to know they
    // are signed, which $signed() guarantees.
    wire signed [15:0] p0 = a0 * b0;
    wire signed [15:0] p1 = a1 * b1;
    wire signed [15:0] p2 = a2 * b2;
    wire signed [15:0] p3 = a3 * b3;

    // Widen each product to the full accumulator width BEFORE summing.
    //
    // Verilog would actually get this right on its own: expression width is
    // determined by the assignment context, so the additions below would be
    // performed in 32 bits even if the operands were left at 16. The
    // all -128 test case (4 * 16384 = 65536, which needs 17 bits) passes
    // either way and proves it.
    //
    // We widen explicitly anyway, for two reasons. First, relying on
    // context-determined width is one of Verilog's genuinely surprising
    // rules, and a reader should not have to know it to trust this line.
    // Second, if someone later factors the sum into an intermediate
    // 16-bit wire, the implicit version would silently start truncating --
    // and the failure would be data-dependent. Verilator flags the implicit
    // form (WIDTHEXPAND) for exactly this reason.
    // {{16{p[15]}}, p} replicates the sign bit 16 times and appends the
    // product -- the same explicit sign-extension idiom used for the int4
    // nibbles above. A bare assignment would also work, but this states the
    // intent in the code rather than leaving it to Verilog's width rules.
    wire signed [31:0] p0_ext = {{16{p0[15]}}, p0};
    wire signed [31:0] p1_ext = {{16{p1[15]}}, p1};
    wire signed [31:0] p2_ext = {{16{p2[15]}}, p2};
    wire signed [31:0] p3_ext = {{16{p3[15]}}, p3};

    // Balanced tree ((p0+p1)+(p2+p3)) rather than a chain, because a tree has
    // depth log2(4)=2 instead of 3 -- a shorter critical path for free.
    wire signed [31:0] sum_comb = (p0_ext + p1_ext) + (p2_ext + p3_ext);

    // Optional pipeline register (see PARAMETERS above).
    reg signed [31:0] sum_q;
    reg               sum_q_valid;

    wire signed [31:0] dot_result = (PIPELINE == 0) ? sum_comb : sum_q;

    //-----------------------------------------------------------------------
    // 4. The accumulator
    //-----------------------------------------------------------------------
    reg signed [`ACCEL_ACC_W-1:0] acc;
    assign acc_value = acc;

    //-----------------------------------------------------------------------
    // The one-shot guard -- DO NOT REMOVE
    //-----------------------------------------------------------------------
    // PicoRV32 holds pcpi_valid high for SEVERAL cycles: it raises valid,
    // then keeps it high until it observes pcpi_ready. Because our response
    // is registered, the core sees ready one cycle after we decide to send
    // it -- so there is at least one cycle where valid is still high and we
    // have already answered.
    //
    // Without a guard, the "if (claimed)" body below would execute on EVERY
    // one of those cycles. For DOT4 that is harmless (rd is rewritten with
    // the same value). For DOT4A it is a silent, data-corrupting bug: the
    // accumulator is incremented twice per instruction, so a dot product
    // over N terms comes out as 2x the correct answer. That is exactly the
    // kind of fault that looks like "the model is just inaccurate" rather
    // than "the hardware is broken".
    //
    // This flag makes every instruction take effect EXACTLY ONCE. It is set
    // when we act and cleared when the core drops valid, i.e. when the
    // instruction is genuinely finished.
    //
    // Regression test: sim/tb_dot4.v holds pcpi_valid high for a long time
    // and asserts the accumulator advanced by exactly one dot product.
    reg responded;

    //-----------------------------------------------------------------------
    // 5. Sequential logic: handshake + state update
    //-----------------------------------------------------------------------
    // "always @(posedge clk)" describes a block of flip-flops: everything
    // assigned with <= inside updates simultaneously at the instant the
    // clock rises. This is the big mental shift from C -- the statements are
    // NOT executed in order; they all take effect at once. See
    // docs/VERILOG_PRIMER.md, "blocking vs non-blocking".
    //
    // Reset is ACTIVE LOW (resetn) to match PicoRV32. resetn==0 means "in
    // reset". Getting this backwards leaves the design permanently held in
    // reset and nothing ever happens, which looks identical to a dead clock.
    always @(posedge clk) begin
        if (!resetn) begin
            acc         <= 0;
            pcpi_wr     <= 1'b0;
            pcpi_rd     <= 32'b0;
            pcpi_ready  <= 1'b0;
            pcpi_wait   <= 1'b0;
            sum_q       <= 0;
            sum_q_valid <= 1'b0;
            responded   <= 1'b0;
        end else begin
            // Default every handshake output low each cycle, then override
            // below. This "default then override" style is what guarantees
            // pcpi_ready is a single-cycle PULSE and not a stuck-high signal
            // -- a stuck pcpi_ready makes the core accept garbage results
            // for later instructions.
            pcpi_wr    <= 1'b0;
            pcpi_rd    <= 32'b0;
            pcpi_ready <= 1'b0;
            pcpi_wait  <= 1'b0;

            if (claimed && !responded) begin
                if (PIPELINE == 0) begin
                    //--------------------------------------------------------
                    // 1-cycle path: the combinational result is already
                    // valid this cycle, so we answer immediately. wait stays
                    // low; ready pulses high for exactly one cycle.
                    //--------------------------------------------------------
                    pcpi_ready <= 1'b1;
                    responded  <= 1'b1;   // act exactly once

                    if (is_dot4) begin
                        pcpi_wr <= 1'b1;
                        pcpi_rd <= dot_result;
                    end else if (is_dot4a) begin
                        // Accumulate. rd is NOT written: pcpi_wr stays low,
                        // so the core leaves the destination register alone.
                        acc     <= acc + sum_comb;
                    end else if (is_accrd) begin
                        // Read-and-clear in one instruction. Doing it as two
                        // instructions (read, then clear) would be a race if
                        // an interrupt landed in between.
                        pcpi_wr <= 1'b1;
                        pcpi_rd <= acc;
                        acc     <= 0;
                    end
                end else begin
                    //--------------------------------------------------------
                    // 2-cycle path: cycle 1 registers the products and
                    // asserts wait; cycle 2 asserts ready with the result.
                    //--------------------------------------------------------
                    if (is_accrd) begin
                        // ACCRD does no multiplying, so it never needs the
                        // extra cycle. Answering it in one cycle even when
                        // PIPELINE=1 is both correct and faster.
                        pcpi_ready <= 1'b1;
                        responded  <= 1'b1;
                        pcpi_wr    <= 1'b1;
                        pcpi_rd    <= acc;
                        acc        <= 0;
                    end else if (!sum_q_valid) begin
                        // Cycle 1: capture, and tell the core to hold on.
                        sum_q       <= sum_comb;
                        sum_q_valid <= 1'b1;
                        pcpi_wait   <= 1'b1;
                    end else begin
                        // Cycle 2: deliver. sum_q_valid drops so the NEXT
                        // instruction starts from cycle 1 again. Forgetting
                        // to clear this flag is how you get every second
                        // dot product returning the previous answer.
                        sum_q_valid <= 1'b0;
                        pcpi_ready  <= 1'b1;
                        responded   <= 1'b1;
                        if (is_dot4) begin
                            pcpi_wr <= 1'b1;
                            pcpi_rd <= sum_q;
                        end else begin // is_dot4a
                            acc     <= acc + sum_q;
                        end
                    end
                end
            end else if (!pcpi_valid) begin
                // The core has dropped valid: this instruction is genuinely
                // finished, so rearm for the next one. Clearing the guard
                // here (rather than as soon as we answer) is what makes it
                // robust to the core holding valid for any number of cycles.
                responded   <= 1'b0;
                sum_q_valid <= 1'b0;
            end
        end
    end

`ifdef FORMAL_OR_SIM_ASSERTS
    // Simulation-only sanity check: we must never assert wait and ready in
    // the same cycle -- that combination is meaningless to the core.
    always @(posedge clk) begin
        if (resetn && pcpi_wait && pcpi_ready) begin
            $display("ASSERT FAIL %0t: dot4_pcpi drove wait and ready together", $time);
            $fatal(1);
        end
    end
`endif

endmodule
