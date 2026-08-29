# Troubleshooting

Organised by **what you see**, because that is what you have when something
breaks.

---

## Setup

### `iverilog: command not found`
Not installed. macOS: `brew install icarus-verilog`. Ubuntu:
`sudo apt-get install iverilog`. See [GETTING_STARTED.md](GETTING_STARTED.md).

### `ERROR: PicoRV32 submodule is missing`
```bash
git submodule update --init --recursive
```
A plain `git clone` does not fetch submodules. `git clone --recursive` does.

### `ERROR: no RISC-V cross-compiler found`
You probably do not need one yet. The RTL tests and the parity harness run
without it — `sim/gen_smoke_hex.py` hand-assembles test firmware. Install the
toolchain only when you want to compile the real C.

If installed but not found, the `sw/Makefile` looks for several prefixes.
Override it:
```bash
make -C sw CROSS=riscv64-unknown-elf-
```

### `ModuleNotFoundError: No module named 'yaml'`
Harmless. `train/config.py` falls back to a built-in parser. Install PyYAML if
you want the canonical one.

### `ModuleNotFoundError: No module named 'torch'`
Only needed for **training**. Quantization, export, parity, simulation and
analysis all work without it. Use `python3 train/quantize.py --synthetic`.

---

## Simulation

### Simulation ends instantly, or the CPU traps at t ≈ 0
The firmware hex is missing or empty, so memory is `x` and the CPU executes
garbage.
```bash
ls -la sim/build/firmware.hex
python3 sim/gen_smoke_hex.py --test boot -o sim/build/boot.hex
```

### `WARNING: $readmemh: Not enough words in the file`
**Benign.** It means the program is smaller than the RAM. `run_icarus.sh`
filters it. Only worry if the file is *empty*.

### The simulation hangs and prints nothing
Almost always an unmapped memory access: a bus transaction nobody completes
stalls the CPU forever.

Look for `[soc] WARNING unmapped ...` in the output — `soc_top.v` prints the
offending address specifically to turn this from a mystery into a one-liner.
If you see it, an address in `sw/include/accel.h` disagrees with
`rtl/accel_pkg.vh`.

If there is no warning, suspect a firmware infinite loop or a wedged FSM.
Every testbench has a global timeout that will eventually fire with a message.

### `TEST FAILED -- CPU TRAP`
The core hit an illegal or misaligned instruction. Most common causes:
1. **A custom instruction in a build without the coprocessor.** Check
   `ENABLE_DOT4=1`.
2. **A misaligned 32-bit load.** PicoRV32 is built with `CATCH_MISALIGN=1`.
   Do not cast an arbitrary `int8_t*` to `uint32_t*` — use `load4()` from
   `accel.h`, which does a byte-wise load for exactly this reason.
3. A jump into unwritten memory — usually a stack overflow.

### A testbench passes but the SoC misbehaves
This is a real pattern and it cost us a real bug ([D008](DECISIONS.md)).
Unit testbenches often drive an interface **more politely than the real
master** — dropping `valid` promptly, never holding a signal high longer than
necessary. The real core does not.

If a module passes standalone but fails integrated, make the unit testbench
ruder: hold signals longer, add wait states, deassert at awkward times.

### `x` (red) values in the waveform
An unreset register or an undriven wire. `x` propagates — `x + 1` is `x` — so
trace **backwards** to the first `x`, which is the actual culprit.

### Verilator complains but Icarus is happy
Verilator is much stricter, and it is usually right. Width mismatches,
unintended latches and combinational loops are real bugs that Icarus tolerates.
Fix them.

---

## Bit-exactness

### `PARITY FAILED`
**Stop.** Do not run any measurement until this is fixed — the whole project's
ability to distinguish hardware bugs from numerical drift depends on it.

The output names the first mismatch: the stage, the index, expected, actual.
Common causes:

| Symptom | Likely cause |
|---|---|
| Off by exactly 1, only on negative values | Rounding rule diverged — check the negate branch in `quant.c` / `requantize.v` |
| Wildly wrong, huge values | int32 overflow — the intermediate must be int64 |
| Wrong only for some inputs | Lost signedness — check `$signed()` casts |
| Whole layer shifted by a constant | Bias scaling — bias is in *accumulator* units |
| Layer count differs | Generated header topology disagrees with the config |

If you changed one of the three implementations, you must change all three
together: `train/quant_ref.py`, `sw/src/quant.c`, `rtl/requantize.v`.

### Parity passes but accuracy is poor
Then the arithmetic is *correct* and the *model* is the problem: undertrained,
poorly calibrated activation scales, or genuinely too small. Notably,
`quantize.py` warns when activation scales are defaults rather than
calibrated — uncalibrated scales give correct arithmetic on meaningless
numbers.

### The self-test fails
`self-test: harness detects an injected rounding fault` failing means the
harness is **not actually comparing anything**, and every other PASS is
meaningless. Fix this before anything else.

---

## Build

### `ERROR: program does not fit in RAM`
From the `ASSERT` in `sw/link.ld` — deliberately a build-time failure rather
than runtime corruption. Either shrink the model, or raise `LENGTH` in
`link.ld` **and** `MEM_WORDS` in `rtl/soc_top.v` together. Changing only one
produces a binary that does not fit.

### `'/*' within block comment`
A URL or a path containing `/*` inside a C comment. Rephrase it.

### Weights header not found
```bash
make weights
```
`sw/models/model_weights.h` is a generated build product and is gitignored.

---

## Vivado

### Timing not met (negative WNS)
1. Lower the target clock in `sweep/sweep_config.yaml`.
2. Set `PIPELINE=1` in `dot4_pcpi.v` — the built-in escape hatch, which
   registers the products so the multiplies and adder tree no longer share one
   clock period.
3. Reduce `ARRAY_W` — broadcast fanout grows with width.

**Do not report a cycle count from a design that fails timing as if it were
achievable on hardware.**

### DSP count is higher than expected
Something other than the MAC array is inferring multipliers. Check that
`ENABLE_MUL=0` on the PicoRV32 instance — the whole premise is a core with no
hardware multiply.

### Every configuration synthesises identically
The `-generic` flags are not reaching the RTL. Confirm `synth_design` in
`build.tcl` passes them, and check the elaboration `$display` lines report the
geometry you asked for. This failure is dangerous because it produces a *flat
line* that looks like "the factors do not matter".

---

## Analysis

### ANOVA reports `nan` for F and p
Expected and correct for deterministic responses — see
[DECISIONS.md D009](DECISIONS.md). Zero residual variance means the F test is
undefined. The variance decomposition shown instead is exact.

### Error bars extend below zero
A quantity like stall fraction cannot be negative. This means a real
experimental *factor* is being averaged over and its systematic spread is
being drawn as variability. Hold the factor fixed instead (see
[D010](DECISIONS.md)).

### Effects look huge but the data is synthetic
Check for the `SYNTHETIC DATA - NOT A RESULT` stamp on the figure and the
warning from `load_results.py`. Synthetic results exist to test the pipeline,
and their effects were planted deliberately.

---

## Hardware (once it exists)

### Nothing on the serial terminal
1. Baud 115200, 8N1.
2. Correct device? macOS: `ls /dev/tty.usbserial-*`. Linux: `/dev/ttyUSB*`.
3. Is the bitstream loaded? Check the DONE LED.
4. Is LED0 (`trap`) lit? Then the CPU trapped.
5. `CLKS_PER_BIT` must match: 100 MHz / 115200 = 868.

### Garbled serial output
Wrong baud, or `CLKS_PER_BIT` disagrees with the actual clock. Characters
arrive but are nonsense — a classic sampling-rate mismatch.

### PPK2 reads implausibly high current, identical across configurations
You are measuring **whole-board** current, not the FPGA core rail. The
regulators, FTDI bridge and LEDs dominate and swamp every difference. See
[MEASUREMENT_PROTOCOL.md](MEASUREMENT_PROTOCOL.md) §3.

### PPK2 active current is not above idle
The workload did not run inside the measurement window. Use the GPIO trigger,
or raise `--n-inferences` so the active phase is long relative to the window.

---

## When you are truly stuck

1. `make test` — does the foundation still hold?
2. `git status` / `git stash` — did something change that you forgot about?
3. Rebuild from clean: `make clean && make test`.
4. Reduce to the smallest failing case — one array width, one layer, one
   vector.
5. Write it up in `notebook/NOTEBOOK.md`, including what you already ruled
   out. Judges ask what did not work, and a written trail is how that question
   gets a good answer.
