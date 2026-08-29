# RISC-V Inference Accelerator

**An open characterization of the energy / accuracy / area design space for
neural network inference at microcontroller scale.**

Regeneron ISEF 2027 · Embedded Systems (EBED)

---

## The claim, in plain language

A general-purpose processor is built for branching, unpredictable code. Neural
network inference is the opposite: one arithmetic pattern — multiply-accumulate
— repeated millions of times with almost no branching.

Running inference on a plain RISC-V core spends roughly **8–10 instructions of
overhead for every single useful multiply-add**. That overhead is *structural*,
not a compiler deficiency: the instruction set has no way to express "do four
of these at once", so it cannot be optimised away, only architected away.

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
