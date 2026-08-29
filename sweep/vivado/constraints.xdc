# ===========================================================================
# constraints.xdc -- Digilent Arty A7-100T pin and timing constraints
# ===========================================================================
#
# WHAT AN XDC FILE DOES:
#   Verilog says what the logic is. This says where it physically connects and
#   how fast it must run. Two kinds of statement:
#     set_property PACKAGE_PIN  -- which physical ball of the FPGA package a
#                                  port connects to
#     create_clock              -- the timing requirement the tools must meet
#
#   Get a pin wrong and the design builds cleanly and does nothing, because
#   your UART is wired to an unconnected pin. There is no error message for
#   that, which is why these are copied from Digilent's official master XDC
#   for the Arty A7 rather than guessed.
#
# BOARD: Digilent Arty A7-100T, FPGA XC7A100TCSG324-1
#   NOTE: the A7-35T is retired. The 100T has ~240 DSP slices against the
#   35T's 90, which is what makes the 8x8 array (64 DSPs) comfortably
#   feasible and is why the sweep can go that wide at all.
#
# IO standard is LVCMOS33 throughout: the Arty's banks are powered at 3.3 V.
# ===========================================================================

# ---------------------------------------------------------------------------
# Clock -- 100 MHz oscillator on pin E3
# ---------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN E3  IOSTANDARD LVCMOS33 } [get_ports { clk }]

# 10.000 ns period = 100 MHz. run_sweep.py overrides this via build.tcl when
# sweeping the target frequency; this is the default for a standalone build.
create_clock -add -name sys_clk_pin -period 10.000 -waveform {0 5} [get_ports { clk }]

# ---------------------------------------------------------------------------
# Reset -- the CPU RESET button (active low, matching resetn)
# ---------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN C2  IOSTANDARD LVCMOS33 } [get_ports { resetn }]

# ---------------------------------------------------------------------------
# UART -- to the on-board FTDI USB-serial bridge
# ---------------------------------------------------------------------------
# D10 is the FPGA's TX (uart_rxd_out in Digilent's naming: it is the input to
# the USB bridge). Connect a terminal at 115200 8N1 to read it.
set_property -dict { PACKAGE_PIN D10 IOSTANDARD LVCMOS33 } [get_ports { uart_tx_pin }]

# ---------------------------------------------------------------------------
# Trap indicator -> LED0, so a CPU fault is visible without a terminal
# ---------------------------------------------------------------------------
set_property -dict { PACKAGE_PIN H5  IOSTANDARD LVCMOS33 } [get_ports { trap }]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
set_property CFGBVS VCCO        [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# Quad SPI flash settings, so a bitstream can be stored on the board and boot
# without a host attached -- which matters for the energy measurement: a
# tethered JTAG cable adds current that would corrupt the reading.
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4    [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 33     [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE    [current_design]

# ---------------------------------------------------------------------------
# Timing exceptions
# ---------------------------------------------------------------------------
# The reset button is asynchronous and mechanical. Telling the tools not to
# time it prevents a meaningless timing failure on a path that is never
# clocked in a hurry. The design must still hold reset long enough to be
# recognised, which a button press trivially does.
set_false_path -from [get_ports resetn]

# The UART pin changes at 115200 baud -- roughly a thousand times slower than
# the clock. Timing it against a 10 ns period would be a false constraint.
set_false_path -to [get_ports uart_tx_pin]
set_false_path -to [get_ports trap]

# ---------------------------------------------------------------------------
# ENERGY MEASUREMENT NOTE
# ---------------------------------------------------------------------------
# To measure with the Power Profiler Kit II you must isolate the FPGA core
# supply. On the Arty A7 that means cutting/removing the appropriate shunt
# and wiring the PPK2 in series. Do NOT measure the whole board's USB current:
# it includes the FTDI bridge, the regulators and the LEDs, which together
# dwarf the core power and would make every configuration look identical.
# See docs/MEASUREMENT_PROTOCOL.md before touching the board.
