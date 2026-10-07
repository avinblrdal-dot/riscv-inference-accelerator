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

*Redrafted 2026-10-02 for the off-grid handpump application (DECISIONS.md
D023). The previous industrial-monitoring draft is in git history.*

> A RISC-V Accelerator for Multi-Year, Battery-Powered Failure Detection in
> Off-Grid Water Handpumps

Roughly one in three handpumps in sub-Saharan Africa is broken at any given
time, and repairs often wait weeks because no one knows a pump has failed. A
sensor that detects failure on the pump itself could shorten that wait, but
only if it runs for years on a battery with no grid power and no maintenance
visits. Energy per inference decides whether on-device detection is
deployable at all.

A general-purpose processor spends most of that energy on instruction overhead
rather than useful arithmetic. An open-source RISC-V core with no hardware
multiplier was profiled running int8 neural networks: **99.48%** `[M]` of
baseline cycles fell inside multiply-accumulate loops, a theoretical speedup
ceiling of **191×** `[M]`. A custom dot-product instruction and a parameterized
multiply-accumulate array with local buffering were designed in Verilog,
attached through the core's coprocessor interface, and verified bit-exact
against a Python reference.

Across a full-factorial sweep of array width, buffer depth, and precision, the
best array configuration reached **5.07×** `[M]` on a vibration fault
classifier, while the dot-product instruction reached **5.75×** `[M]` on an
anomaly detector, showing the best design depends on workload shape. On an Artix-7 FPGA, inference energy fell from `__` to `__` mJ (`__`×)
at a cost of `__` LUTs. On real pump recordings, the 8,904-parameter on-chip detector reached **75.9%** `[M]` AUC, above the official challenge baseline. At that
energy, a `__` battery would sustain `__` years of monitoring at one inference
every `__`.

*Word count of the quoted draft, title included, not counting the `[M]`
markers: about 249 (updated 2026-10-03 after the AUC sentence was added). The cap is 250, and filled-in blanks add words, so expect
to trim when the numbers arrive.*

---

## What is already measured and belongs in the abstract

| Quantity | Value | Source |
|---|---|---|
| MAC fraction of baseline cycles | 99.48% | RQ1, cycle-accurate sim |
| Amdahl ceiling | 191× | derived from the above |
| Best array geometry | 4 × 4 | 32-config sweep |
| Best buffer depth | 256 words | 32-config sweep |
| Cycle speedup, best config (workload A) | 5.07× | sweep vs baseline |
| Cycle speedup, workload B (DOT4) | 5.75× | RQ5 preliminary, 6 configs (D021); 5.40× with real weights (D025) |
| Anomaly-detection AUC, workload B, int8 | 75.9% | MIMII pump, DCASE 2020 split (D025) |
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
| ~~Detection accuracy~~ FILLED 2026-10-03: 75.9% AUC on MIMII pump (D025). Still open: real handpump data | Real data — weights are currently synthetic. No public handpump dataset is known; MIMII's pump recordings are the closest stand-in (industrial pumps, not handpumps -- say so) |
| Battery-life sentence | Needs the measured energy, a chosen battery, and a chosen inference rate |
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
