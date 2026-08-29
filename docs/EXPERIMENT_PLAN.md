# Experiment plan

The statistical design. Read this before collecting data — deciding the
analysis *after* seeing results is how people accidentally p-hack.

---

## Research questions and hypotheses

| RQ | Question | Hypothesis | Test |
|---|---|---|---|
| **RQ1** | What fraction of cycles/energy does MAC work consume on an unmodified core? | MAC work is 70–90% of cycles, so the Amdahl ceiling is 3–10× | Descriptive: `cycles_mac / cycles_total` from the baseline build |
| **RQ2** | How much of that ceiling does the accelerator recover, and at what area cost? | `dot4` recovers a modest fraction; the array recovers more but costs DSPs | Speedup vs baseline; LUT/DSP from synthesis |
| **RQ3** | Does optimising data movement beat adding compute? | **Buffer depth × array width interaction is large** — buffering matters far more at width 8 than width 1 | Factorial ANOVA, interaction term |
| **RQ4** | How far can precision drop before accuracy degrades? | int4 costs a few points of accuracy but saves ~half the multiplier area and halves operand bandwidth | Main effect of precision on accuracy and area |
| **RQ5** | Do the gains generalize to a different workload? | Direction of effects is preserved; magnitudes differ | Workload × factor interactions |

**RQ3 is the interesting one.** It is an *interaction* hypothesis, and that
dictates the entire design.

---

## Why full factorial and not one-factor-at-a-time

A one-factor-at-a-time study varies array width with buffering fixed, then
buffering with width fixed. It **cannot detect interactions at all** — it
would report two independent main effects and miss the actual finding.

RQ3 says the effect of buffering *depends on* array width. That is precisely
what an interaction term measures. A full factorial gets every interaction
essentially for free relative to the number of runs.

---

## Factors and levels

| Factor | Levels | n |
|---|---|---|
| `array_w` (with `array_h` = `array_w`) | 1, 2, 4, 8 | 4 |
| `wbuf_depth` (= `abuf_depth`) | 0, 64, 256, 1024 | 4 |
| `precision` | 8, 4 | 2 |
| `workload` | A (CNN), B (autoencoder) | 2 |

**Full factorial: 4 × 4 × 2 × 2 = 64 configurations.**

### Deliberate scope limits

- **Square arrays only.** Allowing `array_h ≠ array_w` would give 256 cells.
  `mac_array.v` supports non-square (and it is tested at 2×8), so this is a
  follow-up study, not a limitation of the hardware.
- **`wbuf_depth` = `abuf_depth`.** Varying them independently would give 256
  cells. If the analysis shows a large buffer main effect, splitting them is
  the obvious next experiment.
- **Level 0 is the control**, not a missing value. It is a real, working
  no-buffer accelerator (see [DECISIONS.md D007](DECISIONS.md)).

---

## Replicates — and an important subtlety

`replicates: 5` in `sweep/sweep_config.yaml`, giving **320 runs**.

**RTL simulation is deterministic.** Identical parameters produce identical
cycle counts every time. Replicating a simulation therefore measures *nothing*
and would manufacture a false appearance of precision.

Replicates exist for the **synthesis** numbers. Vivado's placer and router are
seeded heuristics, so LUT counts and especially Fmax genuinely vary between
runs of an identical design. Reporting a single Fmax as exact would be wrong.

Consequences, both implemented:
- `run_sweep.py --dry-run` collapses replicates to 1 and says why.
- `analysis/anova.py` detects zero residual variance and reports an **exact
  variance decomposition instead of F and p**, because the F test is undefined
  when `SS_error = 0`. See [DECISIONS.md D009](DECISIONS.md).

---

## Response variables

| Response | Source | Stochastic? |
|---|---|---|
| `cycles_total`, `cycles_stall`, `macs_done`, `utilisation` | RTL simulation | **No** — deterministic |
| `luts`, `ffs`, `dsps`, `brams`, `fmax_mhz` | Vivado | Yes (seeded placer) |
| `accuracy` | quantized model on held-out data | No, given a fixed seed |
| `energy_per_inference_uj` | PPK2 | Yes (thermal, supply noise) |

---

## Statistical model

For each response *Y*, on each workload separately:

```
Y ~ array_w + wbuf_depth + precision
    + array_w:wbuf_depth          <- THE RQ3 TERM
    + array_w:precision
    + wbuf_depth:precision
    + array_w:wbuf_depth:precision
```

Factors are treated as **categorical**, not continuous. Levels are powers of
two and effects are strongly nonlinear (stalls are zero until bandwidth
saturates, then rise sharply); forcing a linear term would badly misfit.

**Workloads are analysed separately, not pooled.** They differ by ~40× in MAC
count and have different layer structure; pooling would let workload dominate
the variance and mask the factor effects we care about.

---

## Effect size, and why it leads

With 320 rows almost everything reaches p < 0.05, including effects far too
small to care about. So:

> **Report partial η² first, p-value second.**

Conventional thresholds:

| partial η² | Label |
|---|---|
| < 0.01 | negligible |
| 0.01 – 0.06 | small |
| 0.06 – 0.14 | medium |
| > 0.14 | large |

**Pre-registered decision rule for RQ3:** the hypothesis is supported if
`array_w × wbuf_depth` has partial η² ≥ 0.06 **and** survives FDR correction
(where a p-value exists). Stating this *before* looking at real data is what
makes it a test rather than a story.

---

## Multiple comparisons

7 terms × ~5 responses × 2 workloads ≈ 70 tests. At α = 0.05, roughly 3–4
would look significant by chance alone.

**Benjamini–Hochberg FDR control across the whole family**, α = 0.05. FDR
rather than Bonferroni because this is exploratory characterization: we would
rather tolerate a few false positives than miss a real effect. Bonferroni at
70 tests would require p < 0.0007 and would be badly underpowered.

Correction is applied across *all* tests at once, not per table — correcting
per table under-corrects and inflates the false discovery rate.

---

## Validation of the analysis itself

`analysis/anova.py --validate` generates synthetic data with a **deliberately
planted** `array_w × wbuf_depth` interaction and requires the analysis to
recover it. It currently recovers an interaction explaining **41% of the
variance** in stall fraction — larger than either main effect — and separately
exercises the F-test path on a stochastic response.

An ANOVA script that has never been checked against known ground truth is not
evidence of anything. If this validation ever fails, the model is
mis-specified and would miss the same effect in the real sweep.

---

## UPDATE — RQ3 confirmed testable (2026-08-29)

An earlier revision of this section proposed abandoning `wbuf_depth` as a
factor, on the evidence that the accelerator was idle 99.8% of the time and
recorded zero stalls. **That conclusion was wrong and is retracted here.**

The cause was a software defect, not a flaw in the hypothesis: the convolution
loops re-pushed identical weights at every spatial position, so the buffer was
never actually reused (see [DECISIONS.md D017](DECISIONS.md)). With the loops
inverted, buffer depth becomes causal and measurable:

| `wbuf_depth` | Cycles | Correct? |
|---|---|---|
| 0 | 63,540,912 | **no — class 0 vs golden 2** |
| 16 / 64 | 61,563,051 | yes |
| 256 | 52,050,811 | yes |
| 1024 | 52,050,811 | yes |

64 → 256 gives a 15% reduction and then saturates, because conv2's tile needs
72 words. **The saturation point is the layer's working set**, which is a
clean and interpretable result.

Two changes follow, both recorded before any sweep data was collected:
1. The control level is 16 rather than 0, because depth 0 cannot compute a
   correct result ([D016](DECISIONS.md)).
2. Every sweep row records whether the configuration produced the correct
   classification. A configuration that is fast and wrong must never be
   reported as fast.

The methodological lesson is worth stating plainly: **a null result from a
factor means nothing until you have verified the implementation lets that
factor operate.**

---

## Threats to validity

**Internal**
- *Baseline fairness.* The baseline must be a competent implementation
  compiled at `-O2`. An artificially slow baseline inflates every result. See
  the header of `sw/src/nn_baseline.c`.
- *Different work.* If a measured speedup exceeds the Amdahl ceiling, the
  variants are not computing the same thing. `cycle_capture.py` checks this
  explicitly and prints a warning.
- *Bit-exactness.* All comparisons assume identical outputs. `make test` is a
  precondition for any measurement being meaningful.

**Construct**
- *Energy vs cycles.* Cycles are a proxy for energy, and not a perfect one —
  a wider array uses more energy per cycle. This is why energy must be
  **measured**, and why `pareto.py` labels its cycles×area fallback as a proxy
  in every output.
- *Operand-network simplification.* `accel_top.v` replicates a 32-bit buffer
  word across the array for widths > 4. The MAC results are exact, but the
  8-wide configuration is **optimistic about operand delivery**. Flagged in
  [REVIEW.md](REVIEW.md); do not report it as a measured bandwidth result.

**External**
- *Two workloads is not "general".* RQ5's claim must be phrased as "the
  direction of effects is preserved across two structurally different
  workloads", not "this generalizes to all inference".
- *One FPGA family.* Artix-7 only. Results are reported in pJ/MAC partly so
  others can compare on different silicon.

**Statistical**
- *Deterministic responses.* Handled explicitly rather than by faking noise.
- *Synthetic data.* Any row with `accuracy_is_synthetic = True` is excluded
  from reported accuracy. `load_results.py` warns loudly.

---

## Data collection order

1. `make test` — bit-exactness. **Blocking gate.**
2. Baseline cycles → RQ1 → the Amdahl ceiling. Do this early; it bounds
   everything else and may change the plan.
3. `run_sweep.py --dry-run` — cycles and stalls for all 64 configurations.
4. Vivado sweep with 5 replicates — area and timing.
5. Accuracy from the quantized models at both precisions.
6. PPK2 energy for a *subset* — the Pareto-relevant configurations. Measuring
   all 64 by hand is not realistic; measuring the frontier plus the baseline
   is.
7. ESP32-S3 external baseline.

---

## What would falsify each hypothesis

Worth writing down in advance, because a hypothesis that cannot fail is not a
hypothesis.

| RQ | Falsified if... |
|---|---|
| RQ1 | MAC work is < 40% of baseline cycles (the premise is wrong; the accelerator is not worth building) |
| RQ2 | Speedup ≈ 1×, or area cost exceeds the FPGA's DSP budget |
| RQ3 | Interaction partial η² < 0.06 — buffering and width act independently |
| RQ4 | int4 costs > 10 accuracy points (too coarse to be useful), or saves no measurable area |
| RQ5 | Effects reverse direction on workload B |

**Report the falsification if it happens.** A well-measured negative result —
"buffering did not interact with width, contrary to our prediction" — is a
genuine contribution and far better science than quietly changing the
hypothesis.
