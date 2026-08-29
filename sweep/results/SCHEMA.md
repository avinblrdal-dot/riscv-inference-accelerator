# Sweep results CSV schema

One row per **(configuration, workload, replicate)**. The CSV itself is
gitignored because it is regenerable; this schema is tracked so the format is
documented even when no results exist yet.

## Design factors (the independent variables)

| Column | Type | Values | Meaning |
|---|---|---|---|
| `array_w` | int | 1, 2, 4, 8 | MAC array width |
| `array_h` | int | 1, 2, 4, 8 | MAC array height (= width unless overridden) |
| `wbuf_depth` | int | 0, 64, 256, 1024 | Weight buffer depth in 32-bit words. **0 = control case, no local buffer** |
| `abuf_depth` | int | 0, 64, 256, 1024 | Activation buffer depth |
| `precision` | int | 8, 4 | Weight/activation bit width |
| `workload` | str | workload_a, workload_b | Which frozen model |
| `replicate` | int | 0..N-1 | Replicate index |

## Simulation measurements (exact, deterministic)

| Column | Type | Meaning |
|---|---|---|
| `cycles_total` | int | Cycles from start to done |
| `cycles_active` | int | Cycles in which a MAC step retired |
| `cycles_stall` | int | Cycles the array was ready but had no operands. **The RQ3 evidence.** |
| `macs_done` | int | Total multiply-accumulates retired |
| `utilisation` | float | `macs_done / (array_w*array_h*cycles_total)` |
| `stall_fraction` | float | `cycles_stall / cycles_total` |
| `sim_ok` | bool | False if the run timed out or the testbench reported failure |

## Synthesis measurements (from Vivado; empty when `--dry-run`)

| Column | Type | Meaning |
|---|---|---|
| `luts` | int | Lookup tables used |
| `ffs` | int | Flip-flops used |
| `dsps` | int | DSP48 slices used |
| `brams` | float | Block RAM tiles (can be 0.5) |
| `fmax_mhz` | float | Achieved maximum clock = `1000 / (target_period - WNS)` |
| `wns_ns` | float | Worst negative slack. Negative means timing FAILED |
| `timing_met` | bool | True when `wns_ns >= 0` |
| `synth_ok` | bool | False if synthesis errored |

## Accuracy (from the quantized model, not from hardware)

| Column | Type | Meaning |
|---|---|---|
| `accuracy` | float | Validation accuracy of the quantized model, or reconstruction error for workload B |
| `accuracy_is_synthetic` | bool | **True means the model was trained on synthetic data — NOT a result** |

## Energy (empty until the PPK2 exists)

| Column | Type | Meaning |
|---|---|---|
| `energy_per_inference_uj` | float | `TBD_MEASURED` — direct current sensing, not estimated |
| `energy_source` | str | `measured`, `modelled`, or `TBD_MEASURED`. Never blank. |

## Provenance (recorded on every row)

| Column | Type | Meaning |
|---|---|---|
| `git_sha` | str | Commit that produced the row; `-dirty` suffix if the tree was modified |
| `timestamp_utc` | str | ISO 8601 |
| `iverilog_version` | str | Simulator version |
| `vivado_version` | str | Synthesis tool version, or empty |
| `config_sha256` | str | Frozen config hash of the workload |
| `seed` | int | RNG seed |

## Integrity rules

1. A row with `sim_ok=False` or `synth_ok=False` must be **excluded from
   analysis**, not silently treated as zero.
2. `energy_source` is never blank. If energy was modelled rather than
   measured, the analysis must label it as such in every figure.
3. `accuracy_is_synthetic=True` rows may be used to check the *pipeline* but
   must never appear in a reported accuracy figure.
