# ESP32-S3 external baseline (CMSIS-NN)

**Status: skeleton only. Not built, not measured. No board purchased.**

## Why this exists

Comparing our accelerated RISC-V design against our own unaccelerated RISC-V
baseline answers "did the accelerator help?" It does **not** answer the
question a judge or reviewer will actually ask: *"is this better than just
buying a microcontroller?"*

The ESP32-S3 is the honest comparison. It is cheap, widely deployed, has
vector DSP extensions, and runs the same class of workload through CMSIS-NN /
ESP-NN. If our FPGA design loses to a $3 part on energy per inference, that is
a genuine and publishable finding — and far more interesting than a
self-referential speedup number. **Do not quietly drop this comparison if the
result is unflattering.**

## Fairness requirements

The comparison is only meaningful if both sides run the *same work*:

1. **The same frozen model.** Export the identical int8 weights from
   `train/export_weights.py`. Do not retrain for the ESP32.
2. **The same input.** Use the golden test vector from `sim/golden/`.
3. **Bit-exact output.** The ESP32's classification must match the golden
   output. If CMSIS-NN's requantization differs from ours (it may — see
   `docs/DECISIONS.md`, "Rounding rule"), **document the difference** rather
   than adjusting one side to match.
4. **The same measurement method.** PPK2, same protocol, isolated supply rail.
5. **Report energy per inference and pJ/MAC**, not wall-clock time. The parts
   run at different clock rates, so time alone is not comparable.

## Layout (to be created)

```
esp32_baseline/
├── CMakeLists.txt          # ESP-IDF project
├── sdkconfig.defaults      # pinned config, checked in for reproducibility
├── main/
│   ├── CMakeLists.txt
│   ├── main.c              # runs inference, prints cycles over UART
│   └── model_weights.h     # symlink/copy from sw/models/
└── components/esp-nn/      # ESP-IDF managed component
```

## Build (once a board exists)

```bash
idf.py set-target esp32s3
idf.py build flash monitor
```

## What to record

| Metric | How |
|---|---|
| Cycles per inference | `esp_cpu_get_cycle_count()` around the call |
| Energy per inference | PPK2, per `docs/MEASUREMENT_PROTOCOL.md` |
| Flash + RAM footprint | `idf.py size` |
| Predicted class | Must equal the golden output |

Record every number as `TBD_MEASURED` in the results table until it has
actually been taken. Do not fill this in from the ESP32-S3 datasheet — a
datasheet figure is a specification, not a measurement.
