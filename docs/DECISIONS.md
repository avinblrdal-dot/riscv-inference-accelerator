# Design decisions log

Append-only. Every entry records **what** was decided, **why**, and **what it
would cost to reverse**. When a judge asks "why did you do it that way?", the
answer lives here.

Newest at the bottom.

---

## D001 — Use PicoRV32 rather than writing a core

**Decision.** RISC-V core is [PicoRV32](https://github.com/YosysHQ/picorv32),
vendored as a git submodule in `third_party/picorv32`.

**Why.** Reimplementing a known 5-stage pipeline is a solved problem, would
consume most of the timeline, and is not novel. The contribution of this
project is the *characterization of the accelerator design space*, not the
core. Using a well-tested core also means that when something breaks, the bug
is almost certainly in our code.

**Cost to reverse.** High. Everything in `rtl/soc_top.v` and the PCPI
coprocessor assumes PicoRV32's interfaces.

**Constraint it imposes.** PicoRV32 is not pipelined the way a modern core is,
so absolute cycle counts are not comparable to (say) a Cortex-M4. All our
claims must be *relative* — accelerated vs our own baseline — or expressed in
transferable units (pJ/MAC).

---

## D002 — The MAC array is memory-mapped; only `dot4` uses PCPI

**Decision.** `rtl/dot4_pcpi.v` hangs off the coprocessor port. `rtl/accel_top.v`
is a memory-mapped peripheral at `0x4000_0000`.

**Why.** PCPI **stalls the core** until the coprocessor answers. For a 1-cycle
dot product that is the cheapest possible handshake. For an array that runs
for hundreds of cycles per tile it is actively harmful: the core would sit
frozen for the entire duration and we would have gained area for nothing.
Memory-mapped means the core writes a "go" bit and is then free — it can
prepare the next tile, or execute a `WFI` and let the clock gate, which is the
interesting option for an energy project.

**Cost to reverse.** Moderate; `accel_top.v` would need a PCPI wrapper.

---

## D003 — Rounding rule: round half away from zero, symmetric

**Decision.** Requantization uses

```
prod  = acc * M0                       (64-bit)
half  = 1 << (shift - 1)
prod >= 0:  out =  ( prod + half) >> shift
prod <  0:  out = -((-prod + half) >> shift)
```

**Why the negate trick.** An arithmetic right shift rounds toward negative
infinity. The naive `(prod + half) >> shift` therefore rounds +2.5 → +3 but
−2.5 → −2: asymmetric. That injects a small positive DC bias into *every*
layer, which accumulates through the network and costs real accuracy.
Negating first makes rounding symmetric about zero.

**Why not gemmlowp's exact formulation.** TFLite/CMSIS-NN use a
"SaturatingRoundingDoublingHighMul" with a specific nudge. It is harder to
reproduce identically across Python, C *and* Verilog, and reproducing Google's
last bit is not a scientific requirement. What *is* required is that our three
implementations agree with each other and that the rule is written down.

**Consequence to be honest about.** If we ever compare a single output value
directly against a TFLite reference, it may differ by one LSB. The
ESP32-S3 baseline (`measure/esp32_baseline/`) uses CMSIS-NN and may hit this.
**Document the difference rather than adjusting one side to match.**

**Cost to reverse.** Low but disruptive — change three files
(`train/quant_ref.py`, `sw/src/quant.c`, `rtl/requantize.v`) together and
re-run `make test`. Every previously collected number becomes invalid.

---

## D004 — The Python reference depends on numpy only, never PyTorch

**Decision.** `train/quant_ref.py` and `train/verify_parity.py` import numpy
and nothing heavier. PyTorch is needed only to *train*.

**Why.** The bit-exactness harness is the safety net for the entire project.
If proving parity required a 2 GB dependency it would get skipped, and CI
could not run it. It also pins our numerical semantics to our own explicit
integer code rather than to whichever quantization backend a given PyTorch
release happens to use.

**Cost to reverse.** None; this is a constraint we chose to keep.

---

## D005 — Workload B is a fully-connected autoencoder, not a second CNN

**Decision.** Workload B is an FFT-feature autoencoder anomaly detector
(`train/config/workload_b.yaml`).

**Why.** RQ5 asks whether the gains *generalize*. Two CNNs over spectrograms
would share layer types, data layout and arithmetic intensity, so
"it generalizes" would mean almost nothing. The autoencoder has no
convolutions, takes a flat FFT vector rather than a 2-D image, and is trained
unsupervised. Fully-connected layers have long contiguous reductions and low
data reuse; convolutions have short reductions and high reuse. If the
accelerator helps both, the claim has content.

**Alternative considered.** Keyword spotting on Speech Commands. Rejected as
the default because it shares a spectrogram front end with workload A and is
therefore a *weaker* generality test. The config system supports it if the
team wants it as a third workload.

**Cost to reverse.** Low — swap the config file.

---

## D006 — Two buffer modules rather than one parameterized module

**Decision.** `rtl/weight_buffer.v` and `rtl/activation_buffer.v` are separate
files despite being structurally similar.

**Why.** They have different access patterns and will diverge. Weights are
read as random-access tiles reused across input positions. Activations stream,
and a convolution's sliding window wants a small multi-tap read (a line
buffer) that weights never need — hence the second read port on the
activation buffer. Forcing both through one module would mean unused ports on
each instance. Their depths are also swept independently, and separate modules
make the synthesis report legible.

**Cost to reverse.** Low.

---

## D007 — `WBUF_DEPTH = 0` is a real, working control case

**Decision.** Depth 0 builds a buffer module that passes data straight through
with no storage, rather than being an error or an unbuilt configuration.

**Why.** It is the experimental control for RQ3. If depth 0 were subtly
broken, every "buffering helps" result would be an artifact of comparing
against a broken baseline. `sim/tb_weight_buffer.v` therefore tests depth 0
and depth 256 **in the same simulation**.

---

## D008 — PCPI one-shot guard (a bug that was found and fixed)

**Decision.** `rtl/dot4_pcpi.v` carries a `responded` flag so each instruction
takes effect exactly once.

**Why.** PicoRV32 holds `pcpi_valid` high until it observes `pcpi_ready`.
Because our response is registered, there is at least one cycle where valid is
still high and we have already answered. The original code executed the
instruction body on *every* such cycle. For `DOT4` that was harmless (`rd`
rewritten with the same value); for `DOT4A` it silently **doubled every
accumulation**, so `ACCRD` returned 8 where 4 was correct.

**How it was caught.** Not by the unit testbench — that dropped `valid`
promptly and passed. It was caught by running a real program on the real core
in `tb_soc`. The lesson is recorded here because it generalises: *unit tests
that drive an interface more politely than the real master will miss protocol
bugs.*

**Regression.** `sim/tb_dot4.v` now holds `pcpi_valid` high for 8 extra cycles
and asserts the accumulator advanced by exactly one dot product.

---

## D009 — Deterministic responses get no F test

**Decision.** `analysis/anova.py` detects responses with zero residual
variance and reports an exact variance decomposition instead of F and p.

**Why.** RTL simulation is deterministic: identical parameters give identical
cycle counts, so replicates are exact duplicates and `SS_error` is exactly
zero. The classical F test divides by that and is undefined. Two tempting
wrong answers were rejected:

- reporting `F = inf, p = 0` — a statistical fiction;
- adding artificial noise so the test "works" — manufacturing uncertainty that
  does not exist, which is worse.

Effects on a deterministic response are **exact**; they need describing, not
significance-testing. Synthesis responses (Vivado's placer is seeded and
heuristic) are genuinely stochastic and do get the F test.

**Consequence.** Replicates in `sweep_config.yaml` exist for the *synthesis*
numbers. `run_sweep.py --dry-run` collapses replicates to 1 and says so.

---

## D010 — Figures never average over a factor

**Decision.** `analysis/plots.py` holds `precision` fixed rather than
averaging int8 and int4 together.

**Why.** An earlier version averaged over precision within each point and drew
the min–max range as an error bar. That conflated a systematic *factor* with
random *variability*, and the resulting bars extended below zero — implying
negative stall fractions, which are impossible. Error bars now show
replicate-level spread only, which is legitimately zero for deterministic
responses.

---

## D011 — Hand-assembled smoke firmware, so the RTL is testable with no toolchain

**Decision.** `sim/gen_smoke_hex.py` contains a ~40-line RV32I assembler and
emits test firmware directly.

**Why.** Installing a RISC-V cross-compiler is the biggest setup hurdle for a
new team member, and until it is done *nothing* can be simulated — making it
impossible to tell "my toolchain is broken" from "the RTL is broken". With
this, you can clone the repo, install only Icarus Verilog, and immediately
prove the CPU boots, the bus works, the UART transmits, and the custom
instructions execute.

**Cost to reverse.** None; it is additive.

---

## D012 — `quantize.py --synthetic`, so the pipeline is testable before training

**Decision.** Quantization can generate structurally valid random weights
instead of loading a checkpoint.

**Why.** It lets the entire downstream path — export → C header → compile →
simulate → parity — be developed and tested before any training exists and
without PyTorch installed. Shapes and integer scales are realistic; only the
values are meaningless.

**Safeguard.** Every synthetic artifact is tagged `synthetic: true`, carries an
obviously-fake accuracy, and prints a reminder. `analysis/load_results.py`
warns loudly when it loads synthetic rows.

---

## D013 — Bus accept must be one-shot (a bug that produced fast wrong answers)

**Decision.** `accel_top.v` uses a sticky `bus_served` flag so each bus
request takes effect exactly once.

**Why.** `mem_ready` is a single-cycle pulse, but the master may hold
`mem_valid` high after that pulse returns low. The condition
`mem_valid && !mem_ready` therefore becomes true a *second* time and every
side effect fires twice. Observed symptom: each pushed operand was written to
two consecutive buffer addresses and the write pointer advanced by two, so the
activation buffer held `1,1,2,2,3,3,4,4` instead of `1..8`.

**This is the same bug as [D008](#d008), from the same root cause** — a master
holding a request asserted longer than the slave's response pulse. Having hit
it twice in two different interfaces, treat it as the house failure mode:
*any* handshake in this project needs an explicit one-shot guard.

**How it was caught.** Not by any testbench. The full model ran, terminated,
and produced a confident classification that was simply wrong (predicted 1
against a golden 2). It took a purpose-built minimal probe
(`sw/test/accel_probe.c`) that computes the same dot product in plain C and
compares, plus an RTL trace of the operands the array actually consumed.

---

## D014 — Synchronous buffers need a one-cycle control delay

**Decision.** `accel_top.v` delays `arr_clr`, `arr_en` and `res_push` by one
cycle relative to `accel_ctrl`'s address generation.

**Why.** The buffers are synchronous: an address registered in cycle *t*
yields data in cycle *t+1*. But the FSM registers `arr_en` in the same cycle
*t*, so the array accumulated one step ahead of its own data — multiplying
whatever the buffer held from the previous access, and dropping the final
element of every reduction. `res_push` is delayed with it, or the finished
tile would be captured one cycle before the last accumulate lands.

The classic synchronous-memory off-by-one: no hang, no warning, confident
wrong answer.

---

## D015 — The software must match the array's operand contract

**Decision.** `nn_array.c` packs one value per lane *across output channels*,
not four consecutive values of one vector.

**Why.** `accel_top.v` fans buffer words out so that row *i* reads activation
lane `i%4` and column *j* reads weight lane `j%4`. A buffer word therefore
holds one value for each of four different rows/columns at the same *k* — it
is **not** a 4-element slice of one vector, which is what `DOT4` consumes.

The original code packed `pack4(w[k], w[k+1], w[k+2], w[k+3])` with
`K = in_dim/4`, so with `M=N=1` only cell (0,0) was read and the accumulator
became `sum over w of a[4w]*b[4w]` — **every fourth element, three quarters of
the data silently discarded.**

Two further constraints now respected: `K` is tiled to the buffer depth (the
FC layer's `in_dim = 1024` overran a 256-word buffer and wrapped the write
pointer), and the array geometry is **discovered at run time** through
`ACCEL_REG_CONFIG` rather than assumed, because the sweep builds many shapes
from one firmware image.

**Known limitation, to report honestly.** The 32-bit operand path fans out as
lane `j%4`, so only `min(ARRAY_W, 4)` columns are independent. An 8-wide build
does **not** double throughput under this mapping.

---

## TODO_BLOCKED items

These could not be completed and are **not** worked around with invented data.

### TB001 — Dataset URLs unverified
`train/data.py` records landing pages for MIMII (Zenodo record 3384388) and
the CWRU bearing data centre, but **neither has been fetched from this
machine**. Both require licence acceptance and are multi-gigabyte, so bulk
download is intentionally manual. If a link is dead, follow the printed
instructions. `--synthetic` generates physically motivated fake data
(1× shaft tone for imbalance, 2× for misalignment, ball-pass-modulated
ringing for bearing faults) so the pipeline runs offline.

### TB002 — No energy measurements
The Nordic PPK2 has not been purchased. Every energy column is
`TBD_MEASURED`. `measure/energy_model.py` **refuses** to print a lifetime
figure without a supplied inference energy, and labels everything `[MODEL]`.
`analysis/pareto.py` falls back to a clearly-labelled cycles×area proxy.

### TB003 — No synthesis numbers
Vivado is not installed on the development machine, so no LUT/FF/DSP/BRAM or
Fmax figures exist. `sweep/vivado/build.tcl` and the Arty A7-100T constraints
are written but **have never been run**. Treat the first run as bring-up.

### TB004 — No RISC-V cross-compiler on the development machine
`sw/` is written and its host build is clean under `-Wall -Wextra -Werror`,
but the `.hex` firmware has never been cross-compiled. The SoC has been proven
to boot and execute using hand-assembled firmware instead (see D011).

### TB005 — Workload B exceeds on-chip memory at full size
The full 512-dim autoencoder is ~139k weights, which does not fit in 64 KB
alongside activations and stack. `workload_b.yaml` therefore carries a
`deployed_model` block (~8.8k weights) for firmware, keeping the full model as
a host-only accuracy reference. **Report both, and be explicit about which
was measured.**
