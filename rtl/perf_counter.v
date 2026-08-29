//===========================================================================
// perf_counter.v -- Cycle / stall / MAC counters readable from C
//===========================================================================
//
// WHY THIS MODULE IS SCIENTIFICALLY LOAD-BEARING:
//
//   Research question RQ3 asks whether optimising DATA MOVEMENT reduces
//   energy more than adding COMPUTE. To answer that you must be able to say,
//   for a given run, how many cycles the array spent doing useful arithmetic
//   versus how many it spent waiting for operands to arrive.
//
//   A single "total cycles" number cannot distinguish those two. If a
//   configuration is slow, is it because there are too few multipliers, or
//   because the multipliers are starved? Those have opposite fixes. This
//   module is what makes the difference measurable rather than a matter of
//   opinion:
//
//     cycles_total  = everything between start and done
//     cycles_active = a MAC step actually issued
//     cycles_stall  = the array was enabled but could not proceed because
//                     operands were not ready
//     macs_done     = total useful multiply-accumulates retired
//
//   From these:
//     utilisation   = macs_done / (peak_macs_per_cycle * cycles_total)
//     stall_frac    = cycles_stall / cycles_total
//
//   A configuration with high stall_frac is memory-bound; adding array width
//   will NOT help it, and that prediction is directly falsifiable. This is
//   the counter that turns a hand-wave into a hypothesis test.
//
// ALSO USED FOR RQ1 (the Amdahl ceiling):
//
//   Running the plain-C baseline with these counters tells us what fraction
//   of total cycles sits inside the MAC loop. Amdahl's law then bounds the
//   best possible speedup from accelerating only that part:
//
//       max_speedup = 1 / (1 - fraction_accelerated)
//
//   If MACs are 80% of cycles, no accelerator can ever beat 5x overall, no
//   matter how fast the array is. Knowing that number early prevents the
//   team from chasing an impossible target -- and it is a genuinely
//   interesting result in its own right.
//
// A PRACTICAL NOTE ON WIDTH:
//   32 bits at 100 MHz wraps after ~43 seconds. Inference takes
//   milliseconds, so 32 bits is ample per-inference. If you ever batch
//   thousands of inferences into one measurement window, read and reset the
//   counters between batches rather than widening them.
//
// PORTS:
//   run          high while the region of interest is executing
//   mac_issue    number of MACs retired this cycle (0 when idle/stalled)
//   stall        high when the datapath wanted to advance but could not
//   clear        synchronous reset of all counters (write from software)
//   *_o          the counter values, wired to memory-mapped read registers
//===========================================================================

`include "accel_pkg.vh"

module perf_counter (
    input  wire        clk,
    input  wire        resetn,

    input  wire        clear,       // software-triggered reset of counters
    input  wire        run,         // count only while this is high
    input  wire [15:0] mac_issue,   // MACs retired this cycle
    input  wire        stall,       // wanted to advance, could not

    output reg  [31:0] cycles_total_o,
    output reg  [31:0] cycles_active_o,
    output reg  [31:0] cycles_stall_o,
    output reg  [31:0] macs_done_o
);

    always @(posedge clk) begin
        if (!resetn || clear) begin
            cycles_total_o  <= 32'd0;
            cycles_active_o <= 32'd0;
            cycles_stall_o  <= 32'd0;
            macs_done_o     <= 32'd0;
        end else if (run) begin
            // Total advances every cycle in the region of interest.
            cycles_total_o <= cycles_total_o + 32'd1;

            // A cycle is "active" if real work retired, "stalled" if the
            // datapath was blocked. These are mutually exclusive by
            // construction, and deliberately do NOT have to sum to total:
            // a cycle can be neither (for example while the controller is
            // reloading a tile descriptor). Keeping them independent means
            //     cycles_total - cycles_active - cycles_stall
            // reveals control overhead, which is a third cost centre worth
            // seeing rather than hiding inside one of the other two.
            if (mac_issue != 16'd0) begin
                cycles_active_o <= cycles_active_o + 32'd1;
                macs_done_o     <= macs_done_o + {16'd0, mac_issue};
            end else if (stall) begin
                cycles_stall_o  <= cycles_stall_o + 32'd1;
            end
        end
    end

endmodule
