//===========================================================================
// tb_soc.cpp -- Verilator C++ testbench for the whole SoC
//===========================================================================
//
// WHY THIS EXISTS
// ---------------
// Icarus Verilog INTERPRETS Verilog; Verilator COMPILES it into C++ and
// builds a native binary. On this design that is roughly a 50-100x speedup,
// and that difference is not a convenience -- it is the difference between
// the design-space sweep being possible and being impossible.
//
// Measured on the baseline firmware: one inference is ~250 million cycles,
// because rv32i has no hardware multiply and every `a * b` (including the
// address arithmetic inside the convolution loops) becomes a call to
// libgcc's __mulsi3. Under Icarus that is about an hour of wall time. The
// sweep is 64 configurations x replicates. At Icarus speed that is months;
// with this harness it is minutes.
//
// WHAT IT DOES
// ------------
// Drives the clock and reset, decodes the UART pin back into characters the
// way a real serial terminal would, watches for the CPU trap signal, and
// enforces a cycle budget so a hang fails loudly instead of running forever.
//
// USAGE
//   ./sim/run_verilator.sh sim                     # default firmware
//   ./sim/run_verilator.sh sim --max-cycles 5e8
//   obj_dir/Vsoc_top +firmware=sim/build/x.hex
//===========================================================================

#include <verilated.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>

#include "Vsoc_top.h"

// Must match the CLKS_PER_BIT the design was built with (see run_verilator.sh).
// The UART decoder below samples on this cadence; if the two disagree you get
// plausible-looking garbage characters rather than an obvious failure.
static int g_clks_per_bit = 4;

//---------------------------------------------------------------------------
// UART receiver -- the "laptop" end of the serial link
//---------------------------------------------------------------------------
// Serial framing: the line idles high, a falling edge marks the START bit,
// then 8 data bits LSB-first, then a STOP bit. We sample in the MIDDLE of
// each bit rather than at its edge, where the value is least certain.
// Sampling at edges is the classic reason a UART "almost works".
class UartRx {
 public:
    explicit UartRx(int clks_per_bit) : cpb_(clks_per_bit) {}

    // Feed one clock's worth of the tx pin. Returns a character, or -1.
    int step(uint8_t tx) {
        int out = -1;
        switch (state_) {
        case IDLE:
            if (prev_ == 1 && tx == 0) {      // falling edge = start bit
                state_ = START;
                counter_ = 0;
            }
            break;
        case START:
            // Wait to the middle of the start bit and confirm it is still
            // low. A glitch that is high here was never a real start bit.
            if (++counter_ >= cpb_ / 2) {
                if (tx == 0) {
                    state_ = DATA;
                    counter_ = 0;
                    bit_idx_ = 0;
                    shifter_ = 0;
                } else {
                    state_ = IDLE;            // false start, resynchronise
                }
            }
            break;
        case DATA:
            if (++counter_ >= cpb_) {
                counter_ = 0;
                shifter_ |= (tx & 1) << bit_idx_;
                if (++bit_idx_ == 8) state_ = STOP;
            }
            break;
        case STOP:
            if (++counter_ >= cpb_) {
                out = shifter_;
                state_ = IDLE;
                counter_ = 0;
            }
            break;
        }
        prev_ = tx;
        return out;
    }

 private:
    enum State { IDLE, START, DATA, STOP };
    State   state_    = IDLE;
    int     cpb_;
    int     counter_  = 0;
    int     bit_idx_  = 0;
    uint8_t shifter_  = 0;
    uint8_t prev_     = 1;      // line idles high
};

//---------------------------------------------------------------------------

static void usage(const char* prog) {
    std::printf(
        "usage: %s [options]\n"
        "  +firmware=PATH        program image (default sim/build/firmware.hex)\n"
        "  +max_cycles=N         cycle budget before declaring a hang\n"
        "  +expect=STRING        require this string in the UART output\n"
        "  +quiet                suppress the UART output stream\n"
        "  +progress=N           heartbeat every N million cycles (0 = off)\n",
        prog);
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        }
    }

    // Cycle budget. Generous by default: a full baseline inference is ~250M
    // cycles. A budget is still essential -- without one a genuine deadlock
    // runs until someone notices, which on a fast simulator can be a long time.
    uint64_t max_cycles = 2000ULL * 1000ULL * 1000ULL;
    uint64_t progress_m = 25;
    bool quiet = false;
    std::string expect;

    const char* v = nullptr;
    if ((v = Verilated::commandArgsPlusMatch("max_cycles=")) && v[0]) {
        max_cycles = std::strtoull(std::strchr(v, '=') + 1, nullptr, 10);
    }
    if ((v = Verilated::commandArgsPlusMatch("progress=")) && v[0]) {
        progress_m = std::strtoull(std::strchr(v, '=') + 1, nullptr, 10);
    }
    if ((v = Verilated::commandArgsPlusMatch("expect=")) && v[0]) {
        expect = std::strchr(v, '=') + 1;
    }
    if ((v = Verilated::commandArgsPlusMatch("quiet")) && v[0]) {
        quiet = true;
    }
    if ((v = Verilated::commandArgsPlusMatch("clks_per_bit=")) && v[0]) {
        g_clks_per_bit = std::atoi(std::strchr(v, '=') + 1);
    }

    Vsoc_top* dut = new Vsoc_top;
    UartRx    uart(g_clks_per_bit);

    std::string captured;
    uint64_t    cycle = 0;
    bool        trapped = false;
    uint64_t    next_progress = progress_m * 1000000ULL;

    // ---- reset ----------------------------------------------------------
    // Hold reset low for several cycles. PicoRV32 needs a few clocks to
    // initialise its internal state; releasing immediately leaves registers
    // in an undefined state that Verilator (unlike Icarus) will not forgive.
    dut->resetn = 0;
    for (int i = 0; i < 16; i++) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
    }
    dut->resetn = 1;

    std::fprintf(stderr, "[verilator] reset released, budget %llu cycles\n",
                 (unsigned long long)max_cycles);

    // ---- main loop ------------------------------------------------------
    while (!Verilated::gotFinish() && cycle < max_cycles) {
        dut->clk = 0;
        dut->eval();
        dut->clk = 1;
        dut->eval();

        // Sample the UART pin once per clock, exactly as the hardware sees it.
        int ch = uart.step(dut->uart_tx_pin);
        if (ch >= 0) {
            captured.push_back(static_cast<char>(ch));
            if (!quiet) {
                std::fputc(ch, stdout);
                std::fflush(stdout);
            }
        }

        // A trap means an illegal or misaligned instruction. Usually a custom
        // instruction in a build without the coprocessor, or a jump into
        // unwritten memory.
        if (dut->trap && !trapped) {
            trapped = true;
            std::fprintf(stderr,
                         "\n[verilator] CPU TRAP at cycle %llu\n",
                         (unsigned long long)cycle);
            break;
        }

        cycle++;

        if (progress_m && cycle >= next_progress) {
            std::fprintf(stderr, "[verilator] ... %llu M cycles\n",
                         (unsigned long long)(cycle / 1000000ULL));
            std::fflush(stderr);
            next_progress += progress_m * 1000000ULL;
        }
    }

    dut->final();

    // ---- verdict --------------------------------------------------------
    const bool hit_budget = (cycle >= max_cycles);
    int rc = 0;

    std::fprintf(stderr, "\n[verilator] finished after %llu cycles\n",
                 (unsigned long long)cycle);

    if (trapped) {
        std::fprintf(stderr, "TEST FAILED -- CPU trap\n");
        rc = 1;
    } else if (hit_budget) {
        std::fprintf(stderr,
                     "TEST FAILED -- cycle budget exhausted (%llu).\n"
                     "  Either the design is hung, or the workload genuinely\n"
                     "  needs more cycles -- raise +max_cycles= and re-run.\n",
                     (unsigned long long)max_cycles);
        rc = 1;
    } else if (!expect.empty() &&
               captured.find(expect) == std::string::npos) {
        std::fprintf(stderr,
                     "TEST FAILED -- expected string not seen: '%s'\n",
                     expect.c_str());
        rc = 1;
    } else {
        std::fprintf(stderr, "TEST PASSED\n");
    }

    delete dut;
    return rc;
}
