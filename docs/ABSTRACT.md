# ISEF Abstract — working draft

**Status: NOT SUBMITTABLE.** Blanks remain. This file exists so the abstract
lives under version control next to the numbers that fill it, instead of
drifting in a slide deck. ISEF caps the abstract at **250 words** — recount
every time you fill a blank.

Convention used below:

| Marker | Meaning |
|---|---|
| **`[M]`** | Filled from a measurement already in this repo. Traceable to `sweep/results/sweep_results.csv`. |
| **`__`** | Still blank. Needs hardware, Vivado, or real data. Never fill from an estimate. |

---

## Draft

> A RISC-V Accelerator Architecture for Multi-Year Battery-Powered Machine
> Condition Monitoring at the Edge

General-purpose processors are architected for branching, unpredictable code,
while neural inference is a single arithmetic pattern executed at high volume.
This mismatch is structural rather than incidental, and it appears as energy
spent on instruction overhead rather than useful computation. Because energy
per inference determines whether a sensor node can operate unattended, that
overhead sets a limit on where on-device intelligence can be deployed at all.
Machine condition monitoring illustrates the constraint: rotating equipment
across industrial and water infrastructure is largely unmonitored because cost
per monitoring point, driven by maintenance visits, exceeds what most
deployments support.

An open-source RISC-V core was implemented on an Artix-7 FPGA and profiled
running an int8-quantized fault classifier. Baseline profiling showed
**99.48%** `[M]` of cycles fell within multiply-accumulate operations,
establishing a theoretical speedup ceiling of **191×** `[M]`. A custom
dot-product instruction and a **4 × 4** `[M]` accumulator array with local
weight buffering were designed in Verilog and integrated through the core's
coprocessor interface. Energy was measured by direct current sensing rather
than estimated.

The accelerated design reduced inference energy from `__` to `__` mJ (`__`×)
at a cost of `__` additional LUTs and `__` DSP slices, with classification
accuracy unchanged at `__`%. Comparable gains on a second, unrelated workload
(`__`×) indicate the result is not model-specific. Sweeping array width and
quantization width mapped the energy, accuracy, and area tradeoff and
identified `__` as the minimum-energy configuration meeting target accuracy.
Results are reported as energy per operation so they remain applicable beyond
the models tested.

---

## What is already measured and belongs in the abstract

| Quantity | Value | Source |
|---|---|---|
| MAC fraction of baseline cycles | 99.48% | RQ1, cycle-accurate sim |
| Amdahl ceiling | 191× | derived from the above |
| Best array geometry | 4 × 4 | 32-config sweep |
| Best buffer depth | 256 words | 32-config sweep |
| Cycle speedup, best config | 5.07× | sweep vs baseline |
| Fraction of ceiling reached | **2.7%** | 5.07 / 191 |
| Configurations verified correct | 32 / 32 | golden-vector check |

**"Fraction of ceiling reached" is the number to volunteer, not hide.** 2.7%
of the available ceiling sounds bad and is in fact the most scientifically
interesting result in the project: it says the bottleneck is not arithmetic
throughput, and the sweep shows exactly why — the array is idle over 99% of
the run, starved by a 32-bit operand path. A judge who hears that from you
first reads it as insight. The same judge who extracts it from you reads it as
a hole.

## Blanks and what unblocks each

| Blank | Unblocked by |
|---|---|
| Inference energy, before and after | Nordic PPK2 + Arty A7-100T |
| LUT / DSP area cost | Vivado synthesis (`sweep/vivado/build.tcl`, never yet run) |
| Classification accuracy | Real MIMII / CWRU data — weights are currently synthetic |
| Second-workload speedup | Workload B firmware (RQ5, not started) |
| Minimum-energy configuration | Requires the energy column, not the cycle column |

## Rules for filling this in

1. **Never fill a blank from an estimate.** The abstract is where an estimate
   becomes a claim, and this project's whole methodological stance is that
   energy is measured rather than modelled.
2. **Recount the words after every edit.** 250 is a hard cap.
3. **The claim may not grow past the measurement.** Cycles, joules, area, and
   accuracy, on one FPGA, on a bench. Nothing further.
4. **If a blank is still blank at submission, rewrite the sentence** so the
   abstract describes what was actually done. An abstract that promises a
   measurement it does not report is worse than a narrower one that delivers.
