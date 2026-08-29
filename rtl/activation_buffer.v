//===========================================================================
// activation_buffer.v -- Local on-chip storage for activations (input data)
//===========================================================================
//
// RELATIONSHIP TO weight_buffer.v:
//
//   Structurally these two are near-identical single-port RAMs, and it is
//   fair to ask why they are not one module instantiated twice. They are
//   kept separate deliberately, for two reasons:
//
//     1. They have different ACCESS PATTERNS and will diverge. Weights are
//        read as random-access tiles and are reused across many input
//        positions. Activations stream: in a convolution the same input row
//        is read by several adjacent output positions in a sliding window,
//        so an activation buffer wants a small multi-tap read (a "line
//        buffer"), which weights never need. Forcing both through one
//        module would mean a pile of unused ports on each instance.
//     2. Their DEPTHS are swept independently (WBUF_DEPTH vs ABUF_DEPTH),
//        and keeping them separate makes it obvious in the synthesis report
//        which BRAM belongs to which.
//
//   See docs/DECISIONS.md, entry "Two buffer modules rather than one".
//
// THE SLIDING WINDOW, for a beginner:
//
//   A 3x3 convolution centred at pixel 5 reads pixels 4,5,6 (and the rows
//   above and below). The next output, centred at pixel 6, reads 5,6,7. Two
//   of the three values are the SAME. Without a local buffer you fetch each
//   pixel from main memory three times. With one, you fetch it once. That
//   is a 3x reduction in memory traffic for free, and memory traffic is
//   where the energy goes.
//
// PARAMETERS:  DEPTH (0 = control case, no buffer), WIDTH, PRECISION
// PORTS:       same shape as weight_buffer, plus a second read port so the
//              array can read two window taps in one cycle when DEPTH > 0.
//===========================================================================

`include "accel_pkg.vh"

module activation_buffer #(
    parameter DEPTH     = 256,
    parameter WIDTH     = 32,
    parameter PRECISION = 8
) (
    input  wire             clk,
    input  wire             resetn,

    input  wire             wr_en,
    input  wire [15:0]      wr_addr,
    input  wire [WIDTH-1:0] wr_data,

    // Port A -- primary read
    input  wire             rd_en,
    input  wire [15:0]      rd_addr,
    output wire [WIDTH-1:0] rd_data,

    // Port B -- second window tap (tied off when DEPTH == 0)
    input  wire             rd2_en,
    input  wire [15:0]      rd2_addr,
    output wire [WIDTH-1:0] rd2_data,

    input  wire [WIDTH-1:0] bypass_data,
    output reg  [31:0]      fill_count
);

    localparam ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

    generate
        if (DEPTH == 0) begin : g_nobuf
            // CONTROL CASE: no reuse. Both read ports see main memory, which
            // in practice means the controller can only satisfy one of them
            // per cycle -- and that serialisation is precisely the cost we
            // want the experiment to expose.
            assign rd_data  = bypass_data;
            assign rd2_data = bypass_data;

            always @(posedge clk) begin
                if (!resetn) fill_count <= 32'd0;
            end

            wire _unused = &{1'b0, wr_en, wr_addr, wr_data,
                             rd_en, rd_addr, rd2_en, rd2_addr, 1'b0};

        end else begin : g_buf
            reg [WIDTH-1:0] mem [0:DEPTH-1];
            reg             written [0:DEPTH-1];

            reg [WIDTH-1:0] rd_data_q, rd2_data_q;
            assign rd_data  = rd_data_q;
            assign rd2_data = rd2_data_q;

            integer i;
            always @(posedge clk) begin
                if (!resetn) begin
                    fill_count <= 32'd0;
                    rd_data_q  <= {WIDTH{1'b0}};
                    rd2_data_q <= {WIDTH{1'b0}};
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
                    // Two independent synchronous reads. Vivado infers a
                    // true dual-port BRAM from this pattern. Both have the
                    // same one-cycle latency.
                    if (rd_en)  rd_data_q  <= mem[rd_addr [ADDR_W-1:0]];
                    if (rd2_en) rd2_data_q <= mem[rd2_addr[ADDR_W-1:0]];
                end
            end

            wire _unused = &{1'b0, bypass_data, 1'b0};
        end
    endgenerate

    localparam VALUES_PER_WORD = (PRECISION == 4) ? (WIDTH / 4) : (WIDTH / 8);
    localparam CAPACITY_VALUES = DEPTH * VALUES_PER_WORD;

    initial begin
        if (DEPTH == 0)
            $display("[activation_buffer] DEPTH=0 -- CONTROL CASE, no local reuse");
        else
            $display("[activation_buffer] DEPTH=%0d WIDTH=%0d PRECISION=%0d -> capacity %0d values",
                     DEPTH, WIDTH, PRECISION, CAPACITY_VALUES);
    end

endmodule
