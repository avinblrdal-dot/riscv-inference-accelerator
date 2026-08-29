# Lab Notebook

Dated entries, **newest at the top**. Write down what you did, what worked,
what broke, and any numbers you measured — especially the ugly ones. Your
future self writing the paper will thank you.

See [TEMPLATE.md](TEMPLATE.md) for the entry format and why the "what broke"
section matters most.

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
