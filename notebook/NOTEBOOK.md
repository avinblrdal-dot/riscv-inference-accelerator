# Lab Notebook

Dated entries, **newest at the top**. Write down what you did, what worked,
what broke, and any numbers you measured — especially the ugly ones. Your
future self writing the paper will thank you.

See [TEMPLATE.md](TEMPLATE.md) for the entry format and why the "what broke"
section matters most.

---

## 2026-08-29 (late) — int4 path built; full 3-factor sweep complete

**Goal:** Make `precision` a real factor. It was in the experiment plan but had
never been varied — it showed 0 degrees of freedom in the ANOVA, so the sweep
was a 2-factor experiment pretending to be 3-factor.

**What I did:** Quantized the model to int4, exported a separate header and
its own golden vectors, made `sw/Makefile` able to build against either model,
and taught the sweep to match firmware precision to hardware precision (and to
check each run against the *matching* golden, not the int8 one).

**Results — 32 configurations, all verified correct:**

| Term | Variance explained |
|---|---|
| array width | 77.2% |
| buffer depth | 21.0% |
| width x depth | 1.7% |
| precision | 0.01% |

int4 saves 0.43% of cycles — and exactly 300,500 cycles in *every*
configuration, i.e. a fixed operand-transfer saving rather than anything that
scales.

**What broke / open questions:**
- I set up "int4 model on int8 hardware" expecting it to fail as a control. It
  did not: both give class 2. Correct, and I should have predicted it — int4
  values live in [-7,7], which is inside int8's range, so sign-extending the
  low nibble is a no-op. Not a bug; a bad control.
- RQ4 is only partially answered. Cycles: settled (no benefit). Area and
  energy: blocked on Vivado and the PPK2. Accuracy: blocked on real training
  data. Say so explicitly rather than implying int4 was evaluated in full.

**Next step:** workload B firmware for RQ5; DMA to attack the 0.2% array
utilisation.

---
## 2026-08-29 (evening) — Array variant fixed; full sweep run; RQ3 answered

**Who:** project session

**Goal:** Run the design-space sweep.

**What I did:** Ran the `dot4` and `array` firmware for the first time before
wiring the sweep. The array was 5.1x faster and **produced the wrong answer**
(class 1 vs golden 2) — gap G1 landing exactly as the review predicted. Fixed
it, then found the sweep's premise was also broken, fixed that, then swept.

**Bugs found and fixed (all silent, none caught by existing tests):**
1. **Double bus accept** — `mem_ready` is a one-cycle pulse but the CPU holds
   `mem_valid` longer, so every operand was written to two addresses. Same
   root cause as the earlier PCPI bug (D008). This is the project's
   characteristic failure mode.
2. **Synchronous-memory off-by-one** — array enabled one cycle before its data
   arrived.
3. **Operand contract mismatch** — software packed 4 consecutive elements per
   word; the array wants one value per lane across 4 channels. Three quarters
   of the data was discarded.
4. **Loop order defeated the weight buffer** — weights re-pushed at every
   spatial position (64x redundancy on conv2). This made buffer depth look
   inert and nearly caused me to abandon RQ3 as untestable. **The hypothesis
   was fine; my software defeated it.**
5. **k-tile overran small buffers** — hardcoded 64 against a 16-word buffer.
   Caught only because the sweep checks every result against golden.

**Results (all 16 configurations verified correct):**

| Array | buf 16 | buf 64 | buf 256 | buf 1024 | gain |
|---|---|---|---|---|---|
| 1x1 | 96.5M | 92.4M | 85.7M | 85.7M | 11.1% |
| 2x2 | 75.6M | 72.9M | 62.9M | 62.9M | 16.8% |
| 4x4 | 64.3M | 61.7M | **52.1M** | 52.1M | 19.1% |
| 8x8 | 75.6M | 68.4M | 57.6M | 57.6M | 23.8% |

Best: 4x4 with a 256-word buffer, **5.07x** over the 263.7M-cycle baseline.

**RQ3 verdict: NOT SUPPORTED by the pre-registered criterion.** The
interaction is directionally right and monotone (gain doubles from 11.1% to
23.8%), but explains only 1.7% of variance in log-cycles against main effects
of 77% and 21%. Reporting as stated; not moving the threshold after the fact.

**What broke / open questions:**
- 8x8 is SLOWER than 4x4. Only 4 columns are independent (32-bit fan-out), so
  a wider array adds no parallelism but forces draining 64 FIFO entries per
  tile instead of 16. **More compute made it slower** — a good result.
- ANOVA on raw cycles showed the interaction at 0.8%; on log-cycles 1.7%.
  Performance effects are multiplicative, so log is the correct scale. Noted
  in `analysis/anova.py`.
- The array is still busy only ~0.2% of the time. The CPU pushing operands
  one word at a time dominates everything. DMA or burst transfer is the
  obvious next lever.
- `precision` could not be swept: the exported model is int8, so int4
  hardware would mis-decode it. RQ4 needs an int4 model export first.

**Next step:** int4 model export for RQ4; workload B firmware for RQ5.

---
## 2026-08-29 (later) — RQ1 measured; Verilator makes the sweep feasible

**Who:** project session

**Goal:** Cross-compile the firmware for the first time and measure the
baseline MAC-loop fraction (RQ1).

**What I did:**
- Installed Homebrew, Verilator 5.050, and a prebuilt RISC-V toolchain
  (xPack `riscv-none-elf-gcc` 15.2.0 into `~/.local` — no admin rights, which
  matters because this account is not in sudoers).
- Cross-compiled all three firmware variants for the first time (gap G2).
- Wrote a Verilator C++ testbench (`sim/verilator/tb_soc.cpp`) and extended
  `sim/run_verilator.sh` with `build` and `sim` modes.
- Ran the baseline inference to completion.

**Results / numbers:** (MEASURED — cycle-accurate simulation, exact)

| Quantity | Value |
|---|---|
| cycles_total | 263,729,703 |
| cycles_mac | 262,349,024 |
| instructions retired | 53,201,238 |
| **MAC fraction** | **99.48%** |
| **Amdahl ceiling** | **191x** |
| IPC | 0.202 (5.0 cycles/instruction) |
| cycles per MAC | 708 |
| instructions per MAC | 143 |
| time at 100 MHz | 2.64 s per inference |

Predicted class was **2**, which **matches the golden vector** from the Python
reference. The compiled C running on the simulated RISC-V core agrees with the
reference implementation — full-model agreement on hardware, not just in a
host build.

Simulator speed: Verilator ran 263M cycles in **17 seconds** (15.5M cycles/s).
Icarus reached only 240M cycles in **over an hour** (~64k cycles/s). A **~240x
speedup**.

**What broke / open questions:**
- The README claimed "6-10 instructions of overhead per MAC". Measurement says
  **143 instructions and 708 cycles per MAC** — far worse. Cause: rv32i has
  *no multiply instruction at all*, so every `a * b` calls libgcc's
  `__mulsi3`, and the convolution's address arithmetic (`ox*stride`,
  `iy*in_w`, `ky*kw`) adds three more multiplies per inner iteration. The 6-10
  figure would describe a core that *has* hardware multiply. README corrected.
- First cross-compile failed three ways, all real: `__global_pointer$`
  referenced in `start.S` but never defined in `link.ld`; `-nostdlib`
  excluding libgcc so every `__mulsi3` was undefined; and a 32 KB `scratch`
  array all three variants ignore, which blew the RAM budget. The last was
  caught by the `ASSERT` in `link.ld` at build time rather than as runtime
  corruption — the guard earned its keep.
- `tb_soc.v`'s timeout was an `integer` (32-bit signed). A 4e9 ns value
  wrapped **negative**, silently disabling the timeout — exactly the failure
  the timeout exists to catch. Now `time` (64-bit).
- Verilator found a documentation comment that had become a compiler
  directive: any comment whose first word is `verilator` is parsed as a
  pragma, including `//` comments.
- Validity note: the weights are synthetic, but the **cycle counts are still
  valid**. Control flow in these kernels depends only on tensor shapes, never
  on weight values, so timing is data-independent. Accuracy from this model
  remains meaningless; timing does not.

**Next step:** Wire Verilator into `sweep/run_sweep.py`. At Icarus speed the
64-configuration sweep would take months; at Verilator speed, minutes. Then
measure the `dot4` and `array` variants against this baseline.

---
## 2026-08-29 — Repository foundation built and verified

**Who:** project setup session

**Goal:** Build every piece of software and RTL that does not require physical
hardware, and prove as much of it as possible actually runs.

**What I did:**

- Vendored PicoRV32 as a submodule (`third_party/picorv32`).
- Wrote the RTL: `dot4_pcpi.v` (custom instructions), `mac_unit.v` /
  `mac_array.v` (parameterized MAC grid), `weight_buffer.v` /
  `activation_buffer.v` (with a real depth-0 control case), `accel_ctrl.v`
  (tiling FSM), `accel_top.v`, `perf_counter.v`, `requantize.v`, `uart_tx.v`,
  `soc_top.v`.
- Wrote the three-way bit-exactness harness: `train/quant_ref.py` (numpy
  reference), `sw/src/quant.c`, `rtl/requantize.v`, checked by
  `train/verify_parity.py`.
- Wrote the software stack (`sw/`), training/quantization/export pipeline
  (`train/`), sweep automation (`sweep/`), measurement scripts (`measure/`)
  and analysis (`analysis/`).
- Bootstrapped Icarus Verilog 12 from source on a machine with no package
  manager (built `autoconf` 2.71 and `bison` 3.8.2 first — macOS ships bison
  2.3, which is too old for Icarus's grammar).

**Results / numbers:** (all MEASURED, from simulation — not hardware)

- `verify_parity.py`: **10/10 stages pass.**
  - requantize, dot4, saturate: Python == C
  - conv2d (6 shapes), fc (5 shapes), maxpool (3 shapes): Python == C
  - **whole model: 7 layers, 27,652 intermediate values, bit-identical**
  - requantize (2017 vectors) and dot4 (205 vectors, through the real PCPI
    handshake): Python == Verilog
  - self-test: harness correctly detects a deliberately injected rounding fault
- `./sim/run_icarus.sh`: **all pass**, ~871 checks
  - `tb_dot4` 209, `tb_mac_array` 400, `tb_weight_buffer` 262
  - `tb_soc` boots and prints `RVACCEL-BOOT-OK`
  - `tb_soc` custom instructions print `123` (all three subtests)
- `mac_array` verified at 1×1, 2×2, 8×8 and non-square 2×8 — parameter
  propagation confirmed.
- Sweep dry run (workload A) showing the RQ3 mechanism directly:

  | Config | Cycles | Stalls | Utilisation |
  |---|---|---|---|
  | 1×1, no buffer | 1037 | 0 | 0.99 |
  | 1×1, buffer 1024 | 1037 | 0 | 0.99 |
  | 8×8, no buffer | 1028 | 768 | 0.25 |
  | 8×8, buffer 1024 | 260 | 0 | 0.99 |

  Buffering changes nothing at width 1 and gives ~4× at width 8. That is the
  interaction, visible in raw cycle counts before any statistics.
- `anova.py --validate`: recovers a planted interaction explaining **41.2% of
  variance** in stall fraction — larger than either main effect.

**What broke / open questions:**

- **A real bug, found only by integration.** `dot4_pcpi.v` executed each
  instruction on *every* cycle `pcpi_valid` was high. PicoRV32 holds `valid`
  until it sees `ready`, which is ≥1 cycle after we answer — so `DOT4A`
  accumulated **twice** per instruction and `ACCRD` returned 8 where 4 was
  correct. The standalone testbench passed because it dropped `valid`
  promptly; only running a real program on the real core exposed it. Fixed
  with a `responded` one-shot guard; regression added that holds `valid` high
  for 8 extra cycles. Written up as [DECISIONS.md D008](../docs/DECISIONS.md).
  *Lesson: unit tests that drive an interface more politely than the real
  master will miss protocol bugs.*
- **The ANOVA validation failed the first time**, correctly. RTL simulation is
  deterministic, so replicates are identical, residual variance is exactly
  zero, and F is undefined (`nan`). Rather than injecting fake noise to make
  the test "work", deterministic responses now report an exact variance
  decomposition and omit F/p ([D009](../docs/DECISIONS.md)).
- **The first RQ3 figure had error bars extending to −25% stall fraction**,
  which is impossible. Cause: averaging over `precision`, a real experimental
  factor, and drawing its systematic spread as variability. Figures now hold
  precision fixed ([D010](../docs/DECISIONS.md)).
- Three self-inflicted delays worth recording: wrong UART sampling arithmetic
  in a scratch testbench (250 ns instead of 60 ns); branch offsets off by one
  instruction in hand-written assembly; and a hand-assembled test that
  clobbered its own operand register with the ASCII character it was printing.
  All three looked like RTL bugs at first and were not.
- Energy model, with plausible assumptions, says **sleep leakage is ~89% of
  cycle energy and inference ~1%**. If that survives real measurement, the
  honest headline is "accelerating inference barely moves node lifetime" —
  a system-level negative result that must be reported, not buried.

**Still blocked (no hardware):** PPK2 energy, Vivado area/timing, real MIMII
and CWRU data, cross-compiled firmware. All tracked as `TODO_BLOCKED` in
[DECISIONS.md](../docs/DECISIONS.md).

**Next step:** Install a RISC-V toolchain and cross-compile `sw/`, then get the
baseline MAC-loop fraction (RQ1) — it sets the Amdahl ceiling and bounds
everything else the project can claim.

---

## 2026-07-26 — Repository set up

**Who:** team

**Goal:** Create the project repository and initial structure.

**What I did:**
- Set up repository structure and initial files (README with the scientific
  claim, `.gitignore`, notebook skeleton, empty directory tree).

**Next step:** Get the open-source RISC-V core building on the Artix-7 and run
baseline profiling.

*(Entry carried forward verbatim from the original repository. Preserved
because this notebook is append-only and because it records when the project
actually started.)*
