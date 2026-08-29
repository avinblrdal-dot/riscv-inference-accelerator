# ===========================================================================
# build.tcl -- Non-project-mode synthesis and implementation for the Arty A7
# ===========================================================================
#
# WHY NON-PROJECT MODE:
#   Vivado's GUI "project mode" stores state in a .xpr directory that is
#   binary, machine-specific and hostile to version control. Non-project mode
#   is a plain script: every step is explicit, nothing is hidden, and the same
#   command reproduces the same result on anyone's machine. For a project
#   whose entire claim is reproducibility, that is the only sane choice.
#
# WHAT IT REPORTS:
#   summary.txt with LUT / FF / DSP / BRAM counts and worst negative slack
#   (WNS). sweep/run_sweep.py parses that file. Keep the format stable or the
#   sweep will silently record blanks.
#
# INVOCATION (run_sweep.py does this for you):
#   vivado -mode batch -source build.tcl -tclargs \
#       -part=xc7a100tcsg324-1 -array_h=4 -array_w=4 -wbuf=256 -abuf=256 \
#       -precision=8 -period=10.0 -seed=0 -outdir=... -root=...
# ===========================================================================

# ---- defaults -------------------------------------------------------------
set part      "xc7a100tcsg324-1"
set array_h   4
set array_w   4
set wbuf      256
set abuf      256
set precision 8
set period    10.0
set seed      0
set outdir    "./build"
set root      ".."

# ---- parse -tclargs -------------------------------------------------------
foreach arg $argv {
    if {[regexp {^-(\w+)=(.*)$} $arg -> key val]} {
        switch -- $key {
            part      { set part      $val }
            array_h   { set array_h   $val }
            array_w   { set array_w   $val }
            wbuf      { set wbuf      $val }
            abuf      { set abuf      $val }
            precision { set precision $val }
            period    { set period    $val }
            seed      { set seed      $val }
            outdir    { set outdir    $val }
            root      { set root      $val }
            default   { puts "WARNING: unknown argument '$key' ignored" }
        }
    }
}

file mkdir $outdir
puts "=========================================================="
puts " Synthesising ${array_h}x${array_w} wbuf=${wbuf} p=${precision}"
puts " part=${part} target period=${period} ns  seed=${seed}"
puts "=========================================================="

# ---- read sources ---------------------------------------------------------
# SYNTHESIS is defined so that soc_top.v compiles out its $display/$finish
# simulation hooks. Leaving them in is harmless for Vivado but noisy, and the
# exit port would otherwise look like a real memory region.
set_property verilog_define {SYNTHESIS=1} [current_fileset]

foreach f {dot4_pcpi.v mac_unit.v mac_array.v weight_buffer.v \
           activation_buffer.v accel_ctrl.v accel_top.v perf_counter.v \
           requantize.v uart_tx.v soc_top.v} {
    read_verilog "${root}/rtl/${f}"
}
read_verilog "${root}/third_party/picorv32/picorv32.v"
read_xdc     "${root}/sweep/vivado/constraints.xdc"

# ---- synthesis ------------------------------------------------------------
# The -generic flags are how the sweep's parameters actually reach the RTL.
# If these were dropped, every configuration would synthesise with the
# defaults and the sweep would produce a flat line that looks like "the
# factors do not matter" -- a false negative that would be very hard to spot.
synth_design -top soc_top -part $part \
    -generic ARRAY_H=$array_h \
    -generic ARRAY_W=$array_w \
    -generic WBUF_DEPTH=$wbuf \
    -generic ABUF_DEPTH=$abuf \
    -generic PRECISION=$precision \
    -generic ENABLE_DOT4=1 \
    -generic ENABLE_ACCEL=1

# Create the clock constraint from the requested period.
create_clock -period $period -name sys_clk [get_ports clk]

# ---- implementation -------------------------------------------------------
# The placer and router are heuristic and SEEDED. Different seeds give
# different LUT counts and, especially, different Fmax. That is why
# sweep_config.yaml specifies replicates -- reporting one run's Fmax as if it
# were exact would overstate the precision of the result.
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Default [get_runs impl_1] -quiet

opt_design
place_design -directive Default
phys_opt_design -quiet
route_design -directive Default

# ---- reports --------------------------------------------------------------
report_utilization -file ${outdir}/utilization.rpt
report_timing_summary -file ${outdir}/timing.rpt
report_power -file ${outdir}/power.rpt

# NOTE ON report_power: Vivado's power number is a VECTORLESS ESTIMATE based
# on assumed switching activity. It is NOT a measurement and must never be
# reported as one. It is generated here only as a sanity check -- if the
# estimate and the PPK2 measurement disagree by an order of magnitude,
# something is wrong with one of them. See docs/MEASUREMENT_PROTOCOL.md.

# ---- machine-readable summary --------------------------------------------
set luts  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == LUT}]]
set ffs   [llength [get_cells -hier -filter {PRIMITIVE_GROUP == FLOP_LATCH}]]
set dsps  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == DSP}]]
set brams [llength [get_cells -hier -filter {PRIMITIVE_GROUP == BLOCKRAM}]]

set wns [get_property SLACK [get_timing_paths -delay_type max]]
if {$wns eq ""} { set wns 0.0 }

set fh [open ${outdir}/summary.txt w]
puts $fh "LUT=$luts"
puts $fh "FF=$ffs"
puts $fh "DSP=$dsps"
puts $fh "BRAM=$brams"
puts $fh "WNS=$wns"
puts $fh "ARRAY_H=$array_h"
puts $fh "ARRAY_W=$array_w"
puts $fh "WBUF=$wbuf"
puts $fh "PRECISION=$precision"
puts $fh "PERIOD=$period"
puts $fh "SEED=$seed"
close $fh

puts "Wrote ${outdir}/summary.txt   LUT=$luts FF=$ffs DSP=$dsps BRAM=$brams WNS=$wns"

if {$wns < 0} {
    puts "WARNING: TIMING NOT MET (WNS=${wns} ns)."
    puts "  This configuration cannot run at [expr 1000.0/$period] MHz."
    puts "  Options: lower the clock, or set PIPELINE=1 in dot4_pcpi.v."
    puts "  Do NOT report a cycle count from a design that fails timing as if"
    puts "  it were achievable on hardware."
}

# Write a bitstream only for the default configuration; the sweep needs
# numbers, not 64 bitstreams.
if {$array_h == 4 && $array_w == 4 && $wbuf == 256 && $precision == 8} {
    write_bitstream -force ${outdir}/soc_top.bit
    puts "Wrote ${outdir}/soc_top.bit"
}
