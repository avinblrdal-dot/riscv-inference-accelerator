# Measurement protocol

Exactly how every number in this project is obtained. The organising principle:

> **Every reported number carries a label: MEASURED, MODELLED, or ESTIMATED.
> They are never mixed, and the label is never omitted.**

| Label | Meaning | Example |
|---|---|---|
| **MEASURED** | Read off an instrument or an exact counter | PPK2 current; RTL cycle count |
| **MODELLED** | Calculated from measured inputs plus assumptions | Node battery lifetime |
| **ESTIMATED** | A tool's guess, from assumed conditions | Vivado's vectorless power report |

A judge is entitled to ask "did you measure that or calculate it?" The answer
must be immediate and unambiguous — which means the labelling has to be in the
data, not just in someone's memory.

---

## 1. Cycle counts — MEASURED

**What.** Cycles per inference, and cycles spent inside MAC loops.

**How.** Two independent sources, which should agree:
- `rdcycle` / `rdinstret` CSRs read from C (`sw/include/perf.h`);
- free-running counters in `soc_top.v` at `0x2000_0000`.

**Procedure.**
1. Build the variant: `make -C sw baseline` (or `dot4`, `array`).
2. Run in simulation, or on the board and capture the UART.
3. Parse: `python3 measure/cycle_capture.py --file run.txt --out run.csv`

**Replicates.** In simulation, **one run is sufficient and five are
meaningless** — the simulator is deterministic. On hardware, take 5 runs to
catch nondeterminism from any source you did not anticipate; if they differ,
find out why before proceeding.

**Overhead correction.** Reading the counter costs a few cycles.
`perf_overhead()` measures this at startup and it is reported so it can be
subtracted from short regions.

**Reporting.** Exact integers. No error bars needed in simulation — and adding
them would be dishonest.

**Sanity check.** Baseline speedup must not exceed the Amdahl ceiling implied
by its own MAC fraction. `cycle_capture.py` checks and warns. If it is
exceeded, the variants are not doing the same work — go back to `make test`.

---

## 2. Area and timing — MEASURED (from the tool, on the real netlist)

**What.** LUTs, FFs, DSP slices, BRAM tiles, Fmax.

**How.** Vivado non-project synthesis and implementation
(`sweep/vivado/build.tcl`), parsed from `summary.txt`.

**Procedure.**
```bash
python3 sweep/run_sweep.py            # full sweep with synthesis
```

**Replicates.** **5, and they matter here.** Vivado's placer and router are
seeded heuristics; identical RTL gives different LUT counts and noticeably
different Fmax between runs.

**Reporting.** Mean ± standard deviation across the 5 seeds. **Never report a
single run's Fmax as if it were exact.**

**Fmax derivation.** `Fmax = 1000 / (target_period_ns − WNS_ns)`. If WNS is
negative, timing **failed** and that configuration cannot run at the target
clock. Do not report its cycle count as achievable — either lower the clock
and re-run, or mark it as not meeting timing.

---

## 3. Energy — MEASURED (once the PPK2 exists)

> **STATUS: `TBD_MEASURED`.** The Nordic PPK2 has not been purchased. No energy
> number in this repository is real. `measure/energy_model.py` refuses to print
> a lifetime figure without one, and `analysis/pareto.py` labels its fallback
> as a proxy.

**Why not Vivado's power report.** It is a **vectorless estimate** built on
assumed switching activity and can be off by 2–3× in either direction. It is
generated as a sanity check only. If it and the PPK2 disagree by an order of
magnitude, something is wrong with one of them — investigate rather than
picking the number you prefer.

### Wiring — read before touching the board

**Measure the FPGA core rail, not whole-board USB current.** Board current
includes the regulators, the FTDI USB-serial bridge and the LEDs, which
together dwarf core power. Measured that way, every configuration looks
identical and the experiment silently produces nothing.

1. Identify the FPGA core supply shunt on the Arty A7.
2. Remove/cut it and wire the PPK2 in series, in **ampere-meter** mode (the
   board stays externally powered — the PPK2's source mode cannot supply an
   Artix-7 under load).
3. **Disconnect JTAG during measurement.** The programmer draws current and
   injects noise. Boot from the on-board SPI flash instead — this is why
   `constraints.xdc` sets the SPI configuration properties.

### Procedure

1. **Baseline.** FPGA configured and idle, at least 5 s:
   ```bash
   python3 measure/ppk2_capture.py --port <PORT> --baseline \
       --duration 5 --json-out baseline.json
   ```
2. **Active.** Run N inferences in a loop; N ≥ 100 so the window is long
   compared to the sampling period:
   ```bash
   python3 measure/ppk2_capture.py --port <PORT> --duration 10 \
       --baseline-file baseline.json --n-inferences 100 --out energy.csv
   ```
3. **Repeat 5 times per configuration**, with the board reset between runs.

**Window definition.** Preferred: a GPIO the firmware raises around inference,
captured on the PPK2's digital input. Fallback: the statistical difference
method above.

**Reporting.** Mean ± SD over 5 repeats, in µJ per inference, **and** in
pJ/MAC (divide by the analytic MAC count from `train/models.py`). pJ/MAC is
what makes the result transferable to anyone with a different model and chip —
lead with it.

**Sanity checks.**
- Active current must exceed idle. If not, the workload did not run inside
  the window. `ppk2_capture.py` warns.
- Energy should scale roughly with cycle count within a configuration family.
  A configuration that is 4× faster but uses the same energy is suspicious.

---

## 4. Accuracy — MEASURED (on data), on the quantized model

**What.** Classification accuracy (workload A); reconstruction error and
detection rate at a calibrated threshold (workload B).

**How.** Evaluate the **quantized integer** model — not the float model — on a
held-out split, using `train/quant_ref.py`. The float model's accuracy is
context, not the result.

**Critical rule.** Any artifact tagged `synthetic: true` produces accuracy
that is **NOT A RESULT**. `load_results.py` warns; `pareto.py` warns;
reported figures must exclude it.

**Reporting.** Accuracy at both precisions, with the number of test samples.
For workload B, report the ROC-AUC as well as accuracy at the chosen
threshold, since threshold choice is a free parameter.

---

## 5. Node lifetime — MODELLED, never measured

`measure/energy_model.py`. Every output is prefixed `[MODEL]`, and the script
**refuses to print a lifetime without also printing every assumption**, each
tagged `MEASURED`, `DATASHEET` or `GUESS`.

```bash
python3 measure/energy_model.py --assumptions
python3 measure/energy_model.py --inference-energy-uj <measured> --sensitivity
```

**Always run `--sensitivity` before quoting a lifetime.** The input with the
widest spread is the one worth measuring next, and it is frequently *not*
inference energy — with plausible assumptions, sleep leakage dominates. If
that is what the model says, **report it**: "inference is 1% of the cycle
energy, so accelerating it barely moves deployment lifetime" is an honest and
publishable system-level finding.

---

## 6. External baseline (ESP32-S3) — MEASURED

Same model, same input, same PPK2 protocol. See
[`measure/esp32_baseline/README.md`](../measure/esp32_baseline/README.md).

**Fairness is the whole point.** If the ESP32-S3 wins on energy per inference,
**report that.** It is a more interesting and more credible result than a
self-referential speedup, and hiding it would be misconduct.

---

## 7. Provenance recorded with every number

Every row of `sweep/results/*.csv` carries:

| Field | Why |
|---|---|
| `git_sha` (with `-dirty` flag) | which code produced it |
| `timestamp_utc` | when |
| `iverilog_version`, `vivado_version` | tool versions |
| `config_sha256` | which frozen model |
| `seed` | RNG seed |
| `energy_source` | `measured` / `modelled` / `TBD_MEASURED` — never blank |

A `-dirty` git SHA means the working tree had uncommitted changes. **Treat
such rows as provisional** — they cannot be reproduced exactly.

---

## 8. Integrity rules

1. Rows with `sim_ok = False` or `synth_ok = False` are **excluded**, never
   treated as zeros.
2. `energy_source` is never blank.
3. Synthetic-data rows never appear in a reported accuracy figure.
4. Every figure carries a provenance stamp, so it stays honest when separated
   from its caption — which happens constantly on a poster.
5. `TBD_MEASURED` placeholders are obviously not numbers. Nothing that could
   be mistaken for a measurement is ever written into a results table.
6. If a measurement contradicts the hypothesis, **it goes in the paper**.
