# Glossary

Written for someone who has never touched hardware. If you hit a term in this
repository that is not here, that is a bug — please add it.

---

## Neural network and quantization terms

**MAC (multiply-accumulate)** — the operation `acc = acc + (a × b)`. It is
*the* operation neural network inference is made of. A convolution or a
fully-connected layer is millions of MACs and almost nothing else. Everything
in this project exists to make MACs cheaper.

**Inference** — running an already-trained network on new input to get a
prediction. Distinct from *training*, which is the (far more expensive)
process of learning the weights. We train on a laptop; we do inference on the
FPGA.

**Quantization** — replacing 32-bit floating-point numbers with small integers
(here 8-bit or 4-bit). A trained network holds floats; a microcontroller
cannot afford them. Quantization maps each tensor onto integers via a *scale*
factor: `q = round(x / scale)`.

**int8 / int4** — signed integers in the range −128…127 and −8…7. The whole
inference path uses these instead of floats.

**Requantization** — after a layer accumulates many int8 products into an
int32 accumulator, the result has to be scaled back down to int8 for the next
layer. Doing this with integer multiply-and-shift (never a float multiply) is
what makes bit-exact hardware matching possible. See
[ARCHITECTURE.md](ARCHITECTURE.md).

**Zero point** — the integer value that represents float zero. We use a
*symmetric* scheme where the zero point is 0, so integer zero means float zero
exactly. That matters because padding and ReLU outputs are exactly zero.

**Bit-exact** — two implementations produce *identical bits*, not merely
similar numbers. This project requires PyTorch, C and Verilog to agree
bit-exactly, because without that a hardware bug is indistinguishable from
ordinary numerical drift.

**Golden vectors** — reference inputs and outputs, computed by the trusted
implementation, that every other implementation is checked against.

---

## Hardware and FPGA terms

**RISC-V** — an open instruction set architecture. "Open" means anyone can
build a processor for it without a licence, and — crucially for us — the spec
reserves opcode space for *custom instructions*, which is what lets us add
`DOT4`.

**rv32i** — the 32-bit base integer RISC-V instruction set. Deliberately
minimal: it has **no hardware multiply**. That is not an oversight on our
part; it is the point. It makes the cost of doing MACs in software visible.

**PicoRV32** — the small, open-source RISC-V CPU core we use (as a git
submodule). We did not write it and should not modify it.

**PCPI (Pico Co-Processor Interface)** — PicoRV32's hook for custom
instructions. When the core meets an instruction it does not recognise, it
offers it to an attached coprocessor and waits. See
[`rtl/dot4_pcpi.v`](../rtl/dot4_pcpi.v).

**FPGA (Field-Programmable Gate Array)** — a chip full of generic logic that
you *configure* into whatever circuit you want. Unlike a CPU, you are not
writing a program that runs on fixed hardware; you are describing the hardware
itself. Reconfigurable, so mistakes cost minutes instead of a chip respin.

**Verilog** — the language used to describe hardware. Looks a bit like C.
Behaves nothing like C — see [VERILOG_PRIMER.md](VERILOG_PRIMER.md).

**RTL (Register Transfer Level)** — the abstraction level our Verilog sits at:
describing what happens to data between clock edges, rather than individual
transistors.

**Synthesis** — translating Verilog into a netlist of actual FPGA primitives
(LUTs, flip-flops, DSPs). The hardware equivalent of compiling.

**Place and route** — deciding *where* on the physical chip each piece of
logic goes and *how* the wires connect them. Slow, heuristic, and **seeded** —
which is why running it twice gives slightly different results, and why our
sweep uses replicates.

**LUT (Look-Up Table)** — the FPGA's basic logic building block: a tiny memory
that implements any function of a few inputs. "How many LUTs" is the main
measure of how much logic a design costs.

**FF (Flip-Flop)** — a one-bit memory element that updates on a clock edge.
Registers are made of these.

**DSP slice** — a hardened multiply-accumulate block built into the FPGA.
Using one is far cheaper (in area and power) than building a multiplier out of
LUTs. Our Artix-7 XC7A100T has ~240; an 8×8 int8 MAC array needs 64.

**BRAM (Block RAM)** — dedicated on-chip memory blocks. Our weight and
activation buffers map onto these. Much faster and *far* lower energy per
access than external memory.

**Timing closure** — getting the design to actually run at the target clock
speed. Signals need time to propagate; if the longest path takes longer than
one clock period, the design fails timing and will not work reliably.

**WNS (Worst Negative Slack)** — how much time to spare the slowest path has.
Negative means timing failed.

**Fmax** — the highest clock frequency the design can reliably run at.

**Bitstream** — the configuration file loaded onto the FPGA. The output of the
whole toolchain.

**Testbench** — Verilog that is *not* hardware. It exists only in simulation,
to drive a module's inputs and check its outputs.

---

## Architecture terms

**Systolic array** — a grid of processing elements where data flows
rhythmically between neighbours. Our MAC array is a simpler *broadcast* array;
the term is included because it is what most accelerator papers describe.

**Output stationary** — a dataflow where each cell holds one output element's
accumulator while operands stream past. The alternative, *weight stationary*,
keeps a weight resident instead.

**Arithmetic intensity** — MACs performed per byte of data fetched. The key
number for understanding whether a design is limited by compute or by memory.
An 8×8 array has intensity 64/16 = 4; a 1×1 array has 1/2 = 0.5.

**Memory bound / compute bound** — whether performance is limited by getting
data in, or by doing arithmetic. Most naive accelerators are memory bound, and
adding more multipliers to a memory-bound design does nothing. **This is what
RQ3 tests.**

**Stall** — a cycle where the hardware could do useful work but has no data.
Counting stalls separately from active cycles is how we tell "too few
multipliers" apart from "multipliers are starving".

**im2col** — reshaping a convolution's sliding windows into columns of a
matrix, turning the convolution into a matrix multiply. Costs memory
(overlapping windows duplicate data) but lets one matrix-multiply engine
handle every layer type.

**Memory-mapped** — making hardware look like memory addresses, so ordinary
loads and stores control it. Our MAC array works this way.

**Volatile** (C keyword) — tells the compiler a memory location can change
outside its knowledge, so it must not optimise away reads and writes.
**Forgetting it on a hardware register is the most common bare-metal bug
there is.**

---

## Measurement and statistics terms

**Cycle** — one tick of the clock. At 100 MHz, 10 nanoseconds.

**Joule (J)** — the unit of energy. A microjoule (µJ) is 10⁻⁶ J. Energy, not
power, is what drains a battery: `energy = power × time`.

**pJ/op (picojoules per operation)** — energy per MAC. The unit that makes
results *transferable*: it lets someone with a different model and a different
chip compare against us meaningfully.

**Amdahl's law** — if a fraction *f* of a program's time can be sped up
infinitely, the best possible overall speedup is `1 / (1 − f)`. If MACs are
80% of cycles, no accelerator beats 5×, ever. **This is why RQ1 comes first**:
it sets the ceiling for everything else.

**Duty cycle** — the fraction of time a device is awake. A sensor node might
be awake 0.1% of the time; its battery life is dominated by what it does in
that 0.1% *and* by what it leaks during the other 99.9%.

**ANOVA (Analysis of Variance)** — a statistical method for asking "does this
factor actually affect the outcome, or is the difference just noise?"

**Main effect** — the overall effect of one factor on its own.

**Interaction effect** — when the effect of one factor *depends on* the level
of another. "Buffering helps a lot at width 8 but not at width 1" is an
interaction. **RQ3 is literally an interaction hypothesis**, which is why the
experiment is a full factorial and not one-factor-at-a-time.

**Full factorial** — testing every combination of every factor level. More
runs than changing one thing at a time, but the only way to detect
interactions.

**Effect size (partial η²)** — *how much* a factor matters, as opposed to
whether the effect is statistically detectable. With enough data, tiny
irrelevant effects become "significant"; effect size is what tells you whether
to care.

**p-value** — the probability of seeing an effect this large if the factor
truly did nothing. Small p means "probably not chance". It does **not** mean
"large" or "important".

**Multiple comparisons** — running many tests means some will look significant
by luck. We apply a Benjamini–Hochberg correction across the whole family of
tests.

**Replicate** — repeating a measurement. Note that RTL simulation is
*deterministic*, so replicating it measures nothing; replicates in this project
exist for the *synthesis* numbers, which genuinely vary between tool runs.

**Pareto frontier** — the set of configurations where you cannot improve one
objective without sacrificing another. When objectives conflict (energy vs
accuracy vs area), there is no single "best" — there is a frontier, and
choosing a point on it is an engineering decision, not a data one.

**Control case** — the configuration that deliberately *lacks* the thing being
tested, so you have something to compare against. Here, `WBUF_DEPTH=0`: a real,
working accelerator with no local buffering at all.
