# RISC-V Inference Accelerator

**An open characterization of the energy / accuracy / area design space for
neural network inference at microcontroller scale.**

Regeneron ISEF 2027 · Embedded Systems (EBED)

---

## The claim, in plain language

A general-purpose processor is built for branching, unpredictable code. Neural
network inference is the opposite: one arithmetic pattern — multiply-accumulate
— repeated millions of times with almost no branching.

Running inference on a plain RISC-V core is far more expensive than it looks.
We **measured** it: **143 instructions and 708 cycles per useful
multiply-add** on an rv32i core.

The reason is that rv32i has *no multiply instruction at all*. Every `a * b`
becomes a call into a libgcc software routine, and a convolution's address
arithmetic (`ox*stride`, `iy*in_w`, `ky*kw`) adds three more multiplies per
inner iteration. The overhead is *structural*, not a compiler deficiency: the
instruction set cannot express "do four of these at once", so it cannot be
optimised away — only architected away.

This matters because energy per inference determines whether a battery-powered
sensor can run unattended for years, which determines cost per deployed node,
which determines how many machines get monitored at all. The overhead is a
binding limit on where always-on intelligence can be deployed.

**The contribution is not "we built an accelerator."** Accelerating MACs is
well understood. The contribution is an *open, reproducible characterization*
of the design space at this scale, with energy **measured by direct current
sensing rather than estimated**, validated across two structurally different
workloads, and reported in transferable units (pJ/MAC).

---

## Research questions

| | Question |
|---|---|
| **RQ1** | What fraction of cycles and energy does MAC work consume on an unmodified RISC-V core? *(This sets the Amdahl ceiling for everything else.)* |
| **RQ2** | How much of that ceiling can a custom instruction plus a MAC array actually recover, and at what silicon area cost? |
| **RQ3** | **Does optimising data movement reduce energy more than adding compute units?** *(The most interesting question — it is an interaction effect, and the experiment is designed to test it statistically.)* |
| **RQ4** | How far can precision be reduced (int8 → int4) before accuracy degrades, and how does precision interact with array width and buffer depth? |
| **RQ5** | Do the gains generalize to a structurally different workload? |

**Application anchor:** machine condition monitoring — classifying motor and
pump acoustic/vibration data as `normal`, `imbalance`, `misalignment`, or
`bearing_fault`. Chosen because its energy constraint is the most severe: a
sensor bolted to a pump cannot be recharged.

---

## Results

> **No results yet.** Every entry below is `TBD_MEASURED` because the hardware
> has not been purchased. This project does not substitute estimates for
> measurements — see [MEASUREMENT_PROTOCOL.md](docs/MEASUREMENT_PROTOCOL.md).

| Metric | Baseline | Accelerated |
|---|---|---|
| Inference energy (µJ) | `TBD_MEASURED` | `TBD_MEASURED` |
| Cycles per inference | `TBD_MEASURED` | `TBD_MEASURED` |
| Energy reduction | — | `TBD_MEASURED` |
| Classification accuracy | `TBD_MEASURED` | `TBD_MEASURED` |
| Additional LUTs | — | `TBD_MEASURED` |
| Additional DSP slices | — | `TBD_MEASURED` |
| pJ per MAC | `TBD_MEASURED` | `TBD_MEASURED` |

Blocked on: Nordic PPK2 (energy), Vivado install (area/timing), Arty A7-100T
(both), MIMII/CWRU datasets (accuracy). All tracked in
[DECISIONS.md](docs/DECISIONS.md) under `TODO_BLOCKED`.

### RQ1 — answered (MEASURED, cycle-accurate simulation)

Baseline firmware, workload A, one inference:

| Quantity | Value |
|---|---|
| Total cycles | 263,729,703 |
| Cycles in MAC loops | 262,349,024 |
| **MAC fraction** | **99.48%** |
| **Amdahl ceiling** | **191×** |
| IPC | 0.202 (5.0 cycles/instruction) |
| Cycles per MAC | 708 |
| Time at 100 MHz | 2.64 s per inference |

**99.48% of cycles are MAC work**, so the theoretical ceiling for *any*
accelerator on this workload is 191×. That is an unusually favourable Amdahl
position, and it is a direct consequence of rv32i having no hardware multiply.

The firmware's predicted class matched the Python reference's golden vector
exactly — the compiled C on the simulated core agrees with the reference
implementation.

Cycle counts are exact (simulation is deterministic) and remain valid despite
the model's synthetic weights, because control flow in these kernels depends
only on tensor shapes, never on weight values.

### RQ2 — answered (MEASURED)

All three variants produce the **same, correct** classification (class 2,
matching the golden vector), so these are like-for-like comparisons:

| Variant | Cycles | Speedup |
|---|---|---|
| baseline | 263,729,703 | 1.00× |
| `dot4` | 270,269,701 | **0.98× — slower than baseline** |
| MAC array | 60,453,159 | **4.36×** |

**`dot4` does not help this workload.** A 4-wide SIMD instruction needs four
*contiguous* elements, and a 3×3 convolution kernel has only three per row. So
98.9% of the MACs get no benefit, while the wider code path costs a little
extra. This was predicted in `sw/src/nn_dot4.c` before it was measured, and it
is the direct motivation for the array's im2col approach.

**The array achieves 4.36× — but that is only 2.3% of the 191× ceiling.**
The reason is the interesting part:

| | |
|---|---|
| Accelerator busy | 109,632 cycles |
| Total | 60,453,159 cycles |
| **Array utilisation** | **0.18%** |
| Array internal stalls | **0** |

The array is **idle 99.8% of the time and never stalls internally.** The
bottleneck is not the compute array, and not its local buffers — it is the CPU
packing operands and pushing them one 32-bit word at a time over the memory
bus.

That is a strong early signal for RQ3, and it reframes the question: the
binding constraint on this design is the **CPU-to-accelerator interface**
(DMA, burst transfers, a wider bus), not the array width or buffer depth the
sweep was built to vary. See [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).

### RQ3 / RQ4 — full 3-factor sweep (MEASURED; all 32 configurations verified correct)

Complete factorial: 4 array widths × 4 buffer depths × 2 precisions. Every
configuration was checked against its own golden classification before its
cycle count was recorded.

**Variance in log-cycles explained by each factor:**

| Term | Share | Effect |
|---|---|---|
| array width | **77.2%** | large |
| buffer depth | **21.0%** | large |
| width × depth interaction | 1.7% | small |
| precision | 0.01% | negligible |
| all other interactions | 0.00% | negligible |

**RQ4 (precision), partial answer.** int4 saves **0.43% of cycles** — and
saves *exactly* 300,500 cycles in every single configuration. That is a fixed
reduction in operand-transfer cost, not a scaling effect. Precision is
essentially irrelevant to *runtime* in this design.

That is not the whole of RQ4. int4's real payoff is **area and energy**, which
need Vivado and a power measurement respectively — neither available yet. And
the accuracy half needs a model trained on real data. What we can say now is
narrow but solid: *at this scale, reducing precision does not buy speed.*

### RQ3 — the interaction (MEASURED; all configurations verified correct)

Cycles per inference, workload A, int8. Every cell was checked against the
golden classification before being recorded.

| Array | buf 16 | buf 64 | buf 256 | buf 1024 | Buffer gain |
|---|---|---|---|---|---|
| 1×1 | 96.5M | 92.4M | 85.7M | 85.7M | 11.1% |
| 2×2 | 75.6M | 72.9M | 62.9M | 62.9M | 16.8% |
| 4×4 | 64.3M | 61.7M | **52.1M** | 52.1M | 19.1% |
| 8×8 | 75.6M | 68.4M | 57.6M | 57.6M | 23.8% |

**Best configuration: 4×4 array, 256-word buffer — 5.07× over baseline.**

Three findings:

1. **Array width has an optimum, not a monotone trend.** 8×8 is *worse* than
   4×4. The 32-bit operand path fans out as lane `j mod 4`, so only four
   columns are ever independent — an 8-wide array adds no parallelism but
   forces the CPU to drain 64 FIFO entries per tile instead of 16. More
   compute made it slower.
2. **Buffer depth saturates at the layer's working set.** conv2's tile is 72
   words, so 256 helps and 1024 adds nothing. The saturation point *is* the
   working set, which is a clean and interpretable result.
3. **The interaction is real but small.** Buffer gain rises monotonically with
   width (11.1% → 23.8%, roughly doubling). But on log-cycles it explains only
   **1.7% of variance**, against main effects of 77% (width) and 21% (depth).

**By our pre-registered criterion (partial η² ≥ 0.06), RQ3's interaction
hypothesis is NOT SUPPORTED.** The direction is exactly as predicted and the
trend is monotone, but the magnitude is far below the threshold we committed
to in advance. Buffer depth and array width both matter a great deal, and they
act largely *independently*.

We are reporting that as stated rather than lowering the threshold after
seeing the data. The pre-registration in
[EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) exists for exactly this moment.

### What *is* verified today

These are real, reproducible, and checkable from a clean clone:

| Check | Result |
|---|---|
| Bit-exactness Python == C == Verilog | **10/10 stages pass** |
| — full model, end to end | **7 layers, 27,652 values, bit-identical** |
| — harness self-test (injected fault) | **detected**, as required |
| RTL testbenches | **~871 checks pass** |
| SoC boots and prints over UART | **yes** (`RVACCEL-BOOT-OK`) |
| Custom instructions on the real core | **yes** (DOT4, DOT4A, ACCRD) |
| Parameter propagation | verified at 1×1, 2×2, 8×8, 2×8 |
| Analysis recovers a planted interaction | **41.2% of variance** |

### The RQ3 mechanism, already visible in simulation

Cycle counts from `make sweep-quick` (workload A, deterministic — exact):

| Configuration | Cycles | Stall cycles | MAC utilisation |
|---|---|---|---|
| 1×1, no buffer | 1037 | 0 | 0.99 |
| 1×1, buffer 1024 | 1037 | 0 | 0.99 |
| 8×8, no buffer | 1028 | 768 | 0.25 |
| 8×8, buffer 1024 | **260** | 0 | 0.99 |

Local buffering changes **nothing** at width 1 and gives a **~4× improvement**
at width 8. That is the interaction RQ3 predicts, visible in raw cycle counts
before any statistics. Whether it holds for *energy* is the open question.

---

## Quickstart

```bash
git clone --recursive <your-repo-url>
cd riscv-inference-accelerator

# One package unlocks almost everything:
#   macOS:  brew install icarus-verilog
#   Ubuntu: sudo apt-get install iverilog

make test
```

Expected: `ALL HARDWARE-FREE CHECKS PASSED`.

You need **no FPGA, no RISC-V toolchain, and no PyTorch** for that. See
[GETTING_STARTED.md](docs/GETTING_STARTED.md) for the full tier list.

```bash
make help          # all targets
make parity        # bit-exactness only (the most important check)
make weights       # generate a model, C header and golden vectors
make sweep         # design-space sweep, simulation only
make analysis      # ANOVA, Pareto frontier, figures
```

---

## Repository map

| Path | Contents |
|---|---|
| `rtl/` | Verilog: SoC, custom-instruction coprocessor, MAC array, buffers, counters |
| `sim/` | Testbenches, runner scripts, and a hand-written RV32I assembler for toolchain-free smoke tests |
| `sw/` | Bare-metal C: three inference variants (baseline / dot4 / array), linker script, startup |
| `train/` | Frozen configs, quantization, weight export, **and the bit-exactness harness** |
| `sweep/` | Full-factorial design-space exploration + Vivado TCL flow |
| `measure/` | PPK2 energy capture, UART cycle parsing, node energy model, ESP32-S3 baseline |
| `analysis/` | ANOVA with interaction terms, Pareto frontier, publication figures |
| `docs/` | Architecture, primer, glossary, decisions, protocols, troubleshooting |
| `notebook/` | Dated lab notebook (append-only) |
| `third_party/` | PicoRV32 (submodule — we did not write it) |

### Where to start reading

- New to hardware? → [docs/VERILOG_PRIMER.md](docs/VERILOG_PRIMER.md)
- Unfamiliar term? → [docs/GLOSSARY.md](docs/GLOSSARY.md)
- How does it work? → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Why is it built that way? → [docs/DECISIONS.md](docs/DECISIONS.md)
- Something is broken → [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## Status by phase

| Phase | Status |
|---|---|
| 01 Repo, CI, submodule | Done |
| 02 Simulation infrastructure | Done |
| 03 SoC: CPU + memory + UART + counters | Done — boots and prints |
| 04 Software build system | Done (host build verified; cross-compile untested — no toolchain) |
| 05 Training / quantization / export | Done (synthetic path verified; real training needs PyTorch + datasets) |
| 06 Bit-exactness harness | **Done — 10/10, including a self-test** |
| 07 `dot4` custom instruction | Done — verified on the real core |
| 08 MAC array + buffers + FSM | Done — verified in simulation |
| 09 Vivado flow | Written, **never run** (no Vivado) |
| 10 Sweep automation | Done (dry run); synthesis path untested |
| 11 Measurement scripts | Written, **never run against hardware** |
| 12 Analysis | Done — validated against known ground truth |
| 13 Documentation | Done |
| 14 Energy measurements | **Blocked** — no PPK2 |
| 15 Real datasets | **Blocked** — see `TODO_BLOCKED` |

---

## Scientific integrity

This project is deliberately built so that honesty is enforced by the code
rather than by memory:

- **Every number is labelled** `MEASURED`, `MODELLED`, or `ESTIMATED`. The
  distinction is never blurred.
- **`measure/energy_model.py` refuses** to print a lifetime figure without also
  printing every assumption, each tagged `MEASURED` / `DATASHEET` / `GUESS`.
- **Vivado's power estimate is never reported as energy.** It is a vectorless
  estimate and can be off by 2–3×.
- **Synthetic artifacts are tagged** and warned about at every stage; they can
  exercise the pipeline but never appear as a result.
- **Every figure carries a provenance stamp**, so it stays honest when
  separated from its caption.
- **Failed runs are excluded, never zero-filled.**
- **Deterministic responses get no p-values** — the F test is undefined when
  residual variance is zero, and faking noise to make it "work" would be worse
  than reporting the exact effect.
- **Falsification is pre-registered.** See
  [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) for what would disprove each
  hypothesis. If the energy model's current suggestion holds — that sleep
  leakage dominates and inference is ~1% of node energy — then the honest
  headline is that accelerating inference *barely moves deployment lifetime*.
  That is a real finding and it will be reported.

---

## Reproducing every result

1. `make test` — bit-exactness and RTL, from a clean clone.
2. `make weights` — regenerates the model, C header and golden vectors
   deterministically from the frozen config and its recorded seed.
3. `make sweep` — cycle and stall counts for the design space.
4. `make analysis` — statistics and figures.

Every sweep row records the git SHA, tool versions, frozen config hash and RNG
seed. Configs are hash-locked by `train/freeze.py`, which **refuses** to run if
a frozen config has been edited — so a number in the paper always traces back
to an exact model.

---

## Hardware (not yet purchased)

| Item | Purpose |
|---|---|
| Digilent Arty A7-100T (`XC7A100TCSG324-1`) | ~240 DSP slices, enough for the 8×8 array sweep |
| Nordic Power Profiler Kit II | direct current sensing — measured, not estimated, energy |
| I2S MEMS microphone (INMP441 / SPH0645) | acoustic capture |
| ESP32-S3 | external baseline via CMSIS-NN |

---

## License

MIT — see [LICENSE](LICENSE). PicoRV32 is separately licensed (ISC) by its
authors.
