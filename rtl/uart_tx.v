//===========================================================================
// uart_tx.v -- Transmit-only UART, for getting numbers off the chip
//===========================================================================
//
// WHY A UART AT ALL:
//
//   The FPGA has no screen. When the C program computes "inference took
//   48,213 cycles", that number has to physically leave the chip somehow.
//   A UART is the simplest way: it shifts bytes out on a single wire, one
//   bit at a time, at an agreed rate. A USB-serial chip on the Arty board
//   turns that wire into something a laptop sees as /dev/tty.usbserial.
//
//   In simulation there is no laptop, so sim/tb_soc.v watches this wire,
//   reassembles the bytes, and prints them to the terminal. That is how a
//   printf() in bare-metal C ends up on your screen with no operating
//   system anywhere.
//
// HOW SERIAL FRAMING WORKS:
//
//   The line idles HIGH. To send a byte you drive:
//     1 START bit (low)  -- tells the receiver "a byte is coming"
//     8 DATA bits, least-significant first
//     1 STOP bit (high)  -- returns the line to idle
//   Each bit is held for exactly CLKS_PER_BIT clock cycles. Both ends must
//   agree on that number; if they disagree by more than a few percent the
//   receiver samples at the wrong moments and you get garbage characters.
//   That is what a "wrong baud rate" looks like.
//
// PARAMETERS:
//   CLKS_PER_BIT = clock frequency / baud rate.
//                  100 MHz / 115200 = 868. Must be >= 2.
//
// PORTS:
//   tx_start  pulse high for one cycle with tx_data valid
//   tx_busy   high while a byte is in flight; do not start another
//   tx        the serial output pin
//===========================================================================

module uart_tx #(
    parameter CLKS_PER_BIT = 868
) (
    input  wire       clk,
    input  wire       resetn,
    input  wire       tx_start,
    input  wire [7:0] tx_data,
    output reg        tx,
    output reg        tx_busy
);

    localparam S_IDLE  = 2'd0,
               S_START = 2'd1,
               S_DATA  = 2'd2,
               S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shifter;

    // Held one cycle per bit period.
    wire tick = (clk_cnt == CLKS_PER_BIT - 1);

    always @(posedge clk) begin
        if (!resetn) begin
            state   <= S_IDLE;
            tx      <= 1'b1;      // idle HIGH -- a low line looks like a
                                  // permanent start bit and floods the
                                  // receiver with null bytes
            tx_busy <= 1'b0;
            clk_cnt <= 16'd0;
            bit_idx <= 3'd0;
            shifter <= 8'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx      <= 1'b1;
                    tx_busy <= 1'b0;
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (tx_start) begin
                        shifter <= tx_data;   // capture now; the caller is
                                              // free to change tx_data next
                                              // cycle
                        tx_busy <= 1'b1;
                        state   <= S_START;
                    end
                end

                S_START: begin
                    tx <= 1'b0;               // the start bit
                    if (tick) begin
                        clk_cnt <= 16'd0;
                        state   <= S_DATA;
                    end else clk_cnt <= clk_cnt + 16'd1;
                end

                S_DATA: begin
                    tx <= shifter[0];         // LSB first
                    if (tick) begin
                        clk_cnt <= 16'd0;
                        shifter <= {1'b0, shifter[7:1]};
                        if (bit_idx == 3'd7) begin
                            bit_idx <= 3'd0;
                            state   <= S_STOP;
                        end else bit_idx <= bit_idx + 3'd1;
                    end else clk_cnt <= clk_cnt + 16'd1;
                end

                S_STOP: begin
                    tx <= 1'b1;               // the stop bit
                    if (tick) begin
                        clk_cnt <= 16'd0;
                        tx_busy <= 1'b0;
                        state   <= S_IDLE;
                    end else clk_cnt <= clk_cnt + 16'd1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
