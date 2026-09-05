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

## D016 — The buffer control level is 16, not 0

**Decision.** `sweep_config.yaml` sweeps `wbuf_depth` over `[16, 64, 256,
1024]`. Depth 0 remains supported in the RTL but is excluded from the sweep.

**Why.** Depth 0 builds a buffer with no storage: reads return `bypass_data`,
whatever is on the write bus at that moment. It faithfully models the
*bandwidth* cost of having no local memory — and it is the only configuration
that records nonzero stalls (93,184) — but it **cannot compute a correct
result.** Measured: it returns class 0 against a golden class 2.

Including a functionally broken configuration in a results table would mean
reporting cycle counts for a machine that does not work. Depth 16 is the
smallest depth that computes correctly, so it is a fair minimal-buffering
control.

**Related finding.** Buffer depth has a real, explicable effect once the
software actually reuses weights (see D017): 64 → 256 gives a 15% cycle
reduction and then saturates, because conv2's tile needs 72 words. The
saturation point *is* the layer's working set, which is a clean result.

---

## D017 — Loop order determines whether the weight buffer does anything

**Decision.** `nn_conv2d_array` iterates output-channel groups on the OUTSIDE
and spatial positions on the inside, loading each group's weights once.

**Why.** A convolution applies the same weights at every position — 256 times
for conv2. The original loop order re-pushed identical weights at every
position: 73,728 transfers where 1,152 distinct values exist, a 64x
redundancy (256x for conv1). The buffer was present but never reused.

The consequence was badly misleading. The accelerator sat idle 99.8% of the
time, buffer depth had no measurable effect, and it looked as though RQ3's
hypothesis was wrong. **The hypothesis was fine; the software defeated it.**
Inverting the loops made buffer depth causal — which is what makes RQ3
testable at all.

Worth remembering as a general caution: a null result from a factor is only
meaningful once you have checked that the implementation actually lets that
factor operate.

---

## D018 — Workload B's pipeline scripts must build the `deployed_model`, not `model.layers`

**Decision.** `train/quantize.py::synthetic_model` and `train/export_weights.py`
now branch on `architecture == "fc_autoencoder"` and, when true, build the
topology, input shape and test vector from `cfg["deployed_model"]` instead of
`cfg["model"]["layers"]`.

**Why.** `train/models.py::_build_autoencoder` already made this distinction
correctly for the PyTorch training path (see its own comment), but the two
scripts that generate the firmware header and golden vectors did not. Before
this fix, `quantize.py --synthetic` on workload_b.yaml silently built the
FULL ~139k-weight autoencoder — the one the config's own comment says does
not fit in the SoC's 64 KB of RAM — rather than the ~8.8k-weight
`deployed_model` that is actually meant to ship in firmware. It produced a
header that would either fail to link (too large) or, worse, link against
the wrong model and let a benchmark run without ever noticing it was
measuring something other than what shipped. Caught by trying to actually
build workload B's firmware for RQ5, not by any existing test — nothing was
checking that these two scripts agreed with `models.py` about which model a
`fc_autoencoder` config describes.

---

## D019 — An autoencoder's output is a reconstruction, not a class: `MODEL_TASK`

**Decision.** `export_weights.py` now emits `MODEL_TASK` (`MODEL_TASK_CLASSIFY`
or `MODEL_TASK_RECONSTRUCT`) into the generated header. `sw/src/main.c`
branches on it: a classifier reports `class` against `MODEL_EXPECTED_CLASS`
(unchanged); an autoencoder reports `reconstruction_mae` — the mean absolute
error between the final layer and the original input — against
`MODEL_EXPECTED_RECONSTRUCTION_MAE`, both computed by the same deterministic
integer arithmetic on both sides, so the comparison is bit-exact, not a
tolerance check.

**Why.** Before this, `main.c` unconditionally ran `nn_argmax` on the final
layer and reported it as "class" for every model. For workload A that number
means something. For workload B's autoencoder it is an index into a
reconstructed vector — a number with no meaning, computed and reported as if
it were a result. This is exactly the kind of silent nonsense this project's
bit-exactness discipline exists to prevent (see the project rules on never
reporting a class from a model that was not measuring one). `sweep/run_sweep.py`
was updated to match: its correctness gate now reads `golden["task"]` and
compares `reconstruction_mae` for an autoencoder rather than a `class` field
the firmware never prints for that task — which, unfixed, would have marked
every workload_b sweep row "WRONG ANSWER" even when it was correct, for a
reason having nothing to do with the hardware.

---

## D020 — Verilator's generated Makefile breaks on a space in the repo path

**Decision.** `sweep/run_sweep.py::space_free_root()` detects a space in
`ROOT` and builds through a symlink at a fixed, space-free location
(`/tmp/riscv_inference_accelerator_root`) instead.

**Why.** The repository now lives at
`.../Downloads/Science Fair/riscv-inference-accelerator` — moved there after
this project's Verilator harness was built, and outside this script's
control. Verilator's `--build` step generates a `Vsoc_top.mk` that lists its
source files space-separated and unquoted; a space anywhere in the path
splits one filename into two bogus Make targets, and the build fails with
"No rule to make target '.../Science'" — a message that gives no hint the
real cause is a space three directories up. **This silently blocked every
sweep run from this machine**, for both workloads, not only the new one:
re-running any previously-recorded sweep cell from this location would have
failed the same way before this fix. Confirmed fixed by re-running the
4×4/wbuf=256/int8/workload_a cell and getting the exact previously-recorded
cycle count (52,051,049) back. iverilog and the host `cc` builds are
unaffected — this is specific to Verilator's own generated Makefile, not to
how Python invokes subprocesses.

**How to apply.** If the sweep ever reports a bewildering "No rule to make
target" error again, check whether `ROOT` has grown a space in it before
suspecting the RTL. The symlink workaround makes this transparent going
forward; it does not need to be re-applied by hand.

---

## D021 — RQ5 preliminary timing: MAC fraction and the strategy reversal

**Decision.** With workload B's pipeline fixed (D018, D019) and the sweep
runnable again (D020), a handful of configurations were run to get a first,
honest reading on RQ5 before committing to a full sweep. Recorded here
because the finding changes what the full sweep should prioritize.

**What was measured** (synthetic weights — timing only, not accuracy; see
D012 for why timing remains valid regardless):

| Quantity | Workload A | Workload B |
|---|---|---|
| Baseline MAC fraction | 99.48% | 99.8% |
| Baseline cycles | 263,729,703 | 5,680,598 |
| `dot4` vs baseline | 0.98× (slower) | **5.75×** |
| Best array (4×4/256) vs baseline | 5.07× | 5.33× |
| 8×8 vs 4×4 array | worse | worse (**confirms** D-level finding, not a workload-A fluke) |

**Two findings, and they point in different directions.**

1. **The array-width optimum replicates.** 8×8 underperforming 4×4 was the
   single most surprising result on workload A. Seeing it again, independently,
   on a structurally different model with a different arithmetic profile is
   real evidence it is a property of the operand-delivery path (see the array
   width finding elsewhere in this doc / README), not an artifact of one
   convolution's shape.

2. **The `dot4` vs. array ranking reverses.** On workload A, `dot4` was the
   *worst* variant (slower than doing nothing). On workload B it is the
   *best* — beating even the tuned array. This is not noise: `dot4` needs
   four CONTIGUOUS int8 values to pack into its operand word. A 3×3
   convolution's reduction is only 3 wide, so `dot4` mostly falls back to
   scalar work on workload A. An FC layer's reduction is fully contiguous
   end to end (128, 32, 8, 32 elements), which is exactly `dot4`'s best case.
   This is the mechanistic explanation workload_b.yaml's header predicted in
   advance ("fully-connected layers have long contiguous reductions... if the
   accelerator helps both, the generality claim means something") — it is
   not a post-hoc rationalization.

**What this is not.** Six configurations, not a sweep — no ANOVA, no
variance decomposition, no accuracy (weights are random). RQ5 is not
answered by this entry. It is a reason to run dot4 as a genuine arm of the
full workload-B sweep rather than treating it as workload A's already-settled
loser, and a reason the "does the *strategy that wins* generalize" question
may be more interesting than "does the *speedup number* generalize."

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

### TB004 — RESOLVED 2026-08-29: cross-compiler installed
~~No RISC-V cross-compiler on the development machine.~~ The xPack prebuilt
toolchain is installed at
`~/.local/xpack/xpack-riscv-none-elf-gcc-15.2.0-1/bin/riscv-none-elf-gcc`
and all three firmware variants cross-compile cleanly. Every cycle count in
the sweep comes from real compiled firmware, not hand-assembled stubs. Add the
toolchain to `PATH` before building:

```
export PATH="$HOME/.local/xpack/xpack-riscv-none-elf-gcc-15.2.0-1/bin:$PATH"
```

Kept here rather than deleted so the record shows when the blocker lifted.

### TB005 — Workload B exceeds on-chip memory at full size
The full 512-dim autoencoder is ~139k weights, which does not fit in 64 KB
alongside activations and stack. `workload_b.yaml` therefore carries a
`deployed_model` block (~8.8k weights) for firmware, keeping the full model as
a host-only accuracy reference. **Report both, and be explicit about which
was measured.**

**Status 2026-09-05: the deployed_model path now actually works end to end**
(see D018–D021). Firmware built, bit-exact against the Python reference, and
timing measured on real RTL — but this note about the two different model
sizes still applies whenever a real accuracy number is eventually computed
from the full model on the host and needs to be reported alongside firmware
timing from the deployed one.
