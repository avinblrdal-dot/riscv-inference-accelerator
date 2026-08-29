# Full review

A deliberate audit pass over the whole repository. Written to be useful to the
team rather than reassuring — the gaps and risks sections matter more than the
parts that pass.

**Reviewed at:** repository state after the sweep/measure/analysis work.
**Environment:** macOS, Icarus Verilog 12.0 (built from source), Apple clang,
Python 3.9 + numpy. No Vivado, no RISC-V cross-compiler, no FPGA, no PPK2.

---

## 1. Correctness

### Method

Each module was read for reset behaviour, width handling, signed arithmetic,
indexing, and protocol edge cases, then exercised in simulation. Two bugs were
found and fixed during development; both are recorded here rather than quietly
patched.

### Reset behaviour — PASS

Automated check: every `reg` declared in every RTL file appears inside a
`!resetn` branch. No unreset state.

`resetn` is **active low** throughout, matching PicoRV32. This is the most
likely thing for a newcomer to get backwards, and the symptom (nothing
happens) is indistinguishable from a dead clock — flagged in the primer and
the troubleshooting guide.

`mac_unit.v` deliberately has **no** `else acc <= 0` on its enable path. That
omission is correct and load-bearing: adding one would zero the accumulator
every time the pipeline stalls, silently corrupting long reductions.

### Signed / int8 handling — PASS, and verified by fault injection

This is the classic bug class in this project. Verilog bit-selects are
unsigned by default, so a lost sign is data-dependent and passes roughly half
of any naive test set.

Handled by: explicit `signed` declarations in `mac_unit.v`, `$signed()` casts
in `dot4_pcpi.v`, and the `{{4{x[3]}}, x[3:0]}` sign-extension idiom for int4.

**Verified by deliberately injecting the bug.** Dropping the sign bit on lane 0
of `dot4_pcpi.v` was caught by *both* independent mechanisms:

- `verify_parity.py`: `first mismatch at index 1, inputs a=0x80808080
  b=0x80808080, python expected 65536, Verilog returned 49152`
- `tb_dot4.v`: `FAIL DOT4 all -128 squared: expected 65536, got 49152`

The directed test vectors (`0x80808080`, single-negative-lane) exist precisely
to make this failure loud, and they did.

### Width handling — PASS

- `requantize.v` uses a **64-bit** intermediate. `acc` can reach 2³¹ and `M0`
  is near 2³¹, so a 32-bit product would overflow on nearly every input. This
  is the single most destructive available bug and it is silent.
- Clamping happens in 64-bit **before** narrowing. Narrowing first would wrap
  a large accumulator and then clamp the wrong value.
- `mac_unit.v` sign-extends the 16-bit product to the accumulator width
  explicitly rather than relying on inference.

### Off-by-one and indexing — PASS

- `accel_ctrl.v` uses `>=` rather than `==` for every loop boundary. With
  `==`, a mis-programmed `dim_k = 0` would run 65536 iterations before
  wrapping; with `>=` it exits immediately, and degenerate dimensions route
  straight to `S_DONE`.
- `mac_array.v`'s flat-vector packing (`acc_flat[(gi*ARRAY_W + gj)*ACC_W +: ACC_W]`)
  is the most error-prone expression in the RTL. `tb_mac_array.v` therefore
  checks the **entire output matrix** against an independently computed
  reference — element `[0][0]` is frequently still correct when the packing is
  transposed, so spot-checking would miss it.
- Buffer readback verified across all 256 entries with an address-encoding
  pattern, so a wrong result identifies *which* wrong word was returned.

### PCPI handshake — PASS after fixing a real bug

**Bug found (D008).** `dot4_pcpi.v` originally executed the instruction body on
every cycle `pcpi_valid` was high. PicoRV32 holds `valid` until it *observes*
`ready`, which is at least one cycle after we answer — so `DOT4A` accumulated
**twice per instruction**, and `ACCRD` returned 8 where 4 was correct.

Two things about this are worth internalising:

1. **The unit testbench passed.** It dropped `valid` promptly, more politely
   than the real core does. Only running an actual program on the actual CPU
   exposed it.
2. **The symptom would have looked like poor model accuracy**, not like a
   hardware fault. Every dot product would have been 2× correct — plausible
   garbage.

Fixed with a `responded` one-shot flag; regression test holds `valid` high for
8 extra cycles and asserts the accumulator advanced by exactly one product.

Deadlock safety verified:
- a foreign instruction gets neither `wait` nor `ready` (so the core can trap
  it as illegal) — explicitly tested;
- every claimed instruction asserts `ready` within a bounded number of cycles —
  the testbench's `issue` task fails loudly after 64 cycles rather than hanging;
- `wait` and `ready` are never asserted together (assertion available under
  `FORMAL_OR_SIM_ASSERTS`).

### Deadlock paths outside PCPI — PASS

- `soc_top.v` has an **unmapped-address catch-all** that completes the
  transaction and prints the offending address. Without it, an address typo
  stalls the CPU forever with no output — the most confusing failure mode a
  small SoC has. It must not be deleted.
- `accel_wait_done()` in `accel.h` is bounded, not a bare `while(busy)`. On
  timeout it records the event and `main.c` reports `accel_timeouts`, which
  `cycle_capture.py` treats as invalidating the run.
- Every testbench has a global timeout.

### Known simplifications — not bugs, but must not be misreported

1. **Operand fan-out in `accel_top.v`.** For array widths > 4, a 32-bit buffer
   word is replicated across the array (`(gh % 4)`). The MAC arithmetic is
   exact, but the 8-wide configuration is **optimistic about operand
   delivery**. Do not report an 8×8 result as a measured bandwidth outcome
   without noting this.
2. **Result FIFO drains a whole tile in one cycle.** Real silicon would take
   multiple cycles. It does not affect MAC-cycle accounting, which is what the
   experiment measures.
3. **`accel_ctrl.v` addressing is a simple linear walk**, not strided. The
   experiment measures the *bandwidth ratio*, not the addressing scheme, so
   simplicity here aids interpretability.

---

## 2. Parity — PASS, and the test was tested

### Coverage

| Stage | Coverage |
|---|---|
| Python ↔ C | requantize (2017 vectors), dot4 (505), saturate (208) |
| Python ↔ C, layers | conv2d (6 shapes incl. padding, stride 2, 1×1, even kernels), fc (5 shapes incl. `in_dim=1`), maxpool (3 shapes) |
| Python ↔ C, whole model | **7 layers, 27,652 intermediate values** |
| Python ↔ Verilog | requantize (2017), dot4 (205) through the real PCPI handshake |
| Self-test | injected rounding fault **must** be detected |

Test vectors deliberately include exact-half rounding cases (where the
rounding rule actually bites and which essentially never occur at random),
saturation at both precisions, and the int8 extremes.

### Testing the test — PASS, two independent ways

1. **Built in:** the harness recompiles `quant.c` with the negate-branch
   removed (floor rounding instead of away-from-zero) and **requires** a
   mismatch. It reports `python=[-1,-2,-3,-4] broken_c=[0,-1,-2,-3]` — exactly
   the DC bias the negate trick exists to prevent.
2. **Ad hoc during this review:** a sign-extension fault was injected into the
   *Verilog* and both the parity harness and `tb_dot4` caught it with precise
   first-mismatch diagnostics.

A harness that cannot fail is worse than none, because it produces false
confidence. This one demonstrably fails when it should.

### Gap

`nn_dot4.c` and `nn_array.c` are **not yet** covered by full-model parity —
only `nn_baseline.c` is. They cannot be, without either a RISC-V simulator in
the loop or host stubs for the intrinsics. **This is the single most important
missing test** (see Gap list G1), because the accelerated variants are exactly
where a divergence would matter and would look like an accuracy result.

---

## 3. Parameterization — PASS

Automated check: no sweep parameter is hardcoded anywhere in `rtl/` (the only
matches are in comments).

Propagation verified by building and running four distinct geometries:

| Config | Reported at elaboration | Checks | Result |
|---|---|---|---|
| 1×1 | `built 1x1 ... intensity 1/2` | 25 | pass |
| 2×2 | `built 2x2 ... intensity 4/4` | 100 | pass |
| 8×8 | `built 8x8 ... intensity 64/16` | 1600 | pass |
| 2×8 | `built 2x8 ... intensity 16/10` | 400 | pass |

Every parameterized module prints its built configuration at elaboration.
That is not decoration: when the sweep runs 64 configurations, those lines are
how you confirm a parameter actually reached the RTL instead of silently
defaulting. **A silently defaulted parameter produces a flat line that looks
like "the factors do not matter"** — a false negative that is very hard to
notice. CI checks this explicitly.

Behavioural difference between configurations confirmed (not just elaboration
text): 8×8 unbuffered gives 1028 cycles / 768 stalls, 8×8 buffered gives 260
cycles / 0 stalls.

**Untested:** that `-generic` flags in `build.tcl` actually reach Vivado. If
they do not, the entire area/timing sweep silently produces identical rows.
**Verify this on the very first Vivado run** by diffing two configurations'
`summary.txt`.

---

## 4. Reproducibility — mostly PASS

### What works from a clean clone

`git clone --recursive` then `make test` reproduces every verified result with
one external package (Icarus Verilog). `make weights` regenerates the model,
C header and golden vectors deterministically from the frozen config and its
recorded seed.

Determinism: every RNG is seeded from the frozen config; seeds are recorded in
every artifact; `freeze.py` hash-locks configs and **refuses** to run if one
has been edited (verified — a one-character change to `epochs` is caught).

Provenance on every sweep row: git SHA (with `-dirty` flag), timestamp, tool
versions, config hash, seed, and `energy_source`.

### What would block a stranger

| Blocker | Severity | Mitigation in place |
|---|---|---|
| Datasets require manual download and licence acceptance | Medium | Exact instructions printed; `--synthetic` runs offline |
| No `requirements.lock` — only minimum versions | Low–Medium | Core path needs numpy only |
| Vivado version differences change area/timing numbers | Medium | Version recorded per row |
| PicoRV32 submodule pinned by SHA, not tag | Low | Recorded in `.gitmodules` |
| Icarus built from source in the reference environment | Low | Package-manager install works; documented |

**Recommendation:** generate a `requirements.lock` once the environment
stabilises, and pin the PicoRV32 submodule to a tagged release.

---

## 5. Scientific integrity — PASS

Automated check: no energy figure anywhere in `docs/` or `README.md` is
presented as a result. Every results slot is `TBD_MEASURED`.

Enforcement is structural, not merely conventional:

| Mechanism | Where |
|---|---|
| `energy_model.py` refuses to print a lifetime without an inference energy | exits 1 with instructions |
| It refuses to print a lifetime without also printing every assumption | assumptions tagged `MEASURED`/`DATASHEET`/`GUESS`, guesses counted |
| Vivado power is labelled an estimate at the point of generation | `build.tcl` comment |
| Synthetic artifacts tagged and warned at every stage | quantize → export → load → pareto → figures |
| Failed runs excluded, never zero-filled | `load_results.py`, reported by reason |
| Figures carry a provenance stamp | red text when synthetic or proxy |
| Pareto labels its cycles×area fallback a PROXY | in output and axis label |
| Deterministic responses get no p-values | `anova.py` (D009) |
| Falsification criteria pre-registered | `EXPERIMENT_PLAN.md` |

Two places where the project chose the harder, more honest option rather than
the convenient one:

1. **No fake noise.** RTL simulation is deterministic, so `SS_error = 0` and F
   is undefined. Injecting noise to make the F test "work" would manufacture
   uncertainty that does not exist. Exact variance decomposition is reported
   instead.
2. **The unflattering finding is surfaced, not buried.** With current
   assumptions the energy model says sleep leakage is ~89% of cycle energy and
   inference ~1%, and it prints the interpretation *"even a 10× faster
   accelerator would change lifetime very little"*. That is the kind of result
   a team is tempted to hide; here the tool says it out loud.

**One caution.** The illustrative transcript used to test `cycle_capture.py`
contains invented cycle counts. It is a parser fixture only and appears in no
document — but do not copy those numbers anywhere.

---

## 6. Beginner accessibility

### Strengths

Every module has a concept-first header. `VERILOG_PRIMER.md` leads with the
one genuine mental shift (concurrency) and uses this project's own code.
`GLOSSARY.md` covers ~60 terms. `gen_smoke_hex.py` removes the toolchain from
the critical path entirely — a beginner can prove the CPU boots before
installing a cross-compiler, which separates "my setup is broken" from "the
RTL is broken".

### Jargon appearing before it is explained

Found by re-reading as a newcomer. All are cross-linked to the glossary, but
worth watching:

| Term | Where | Note |
|---|---|---|
| "PCPI" | README status table | glossary-linked, but appears before any explanation |
| "im2col" | `nn_array.c` header | explained in place; unfamiliar on first encounter |
| "Amdahl ceiling" | README results table | explained in `perf.h` and the glossary, not at first use |
| "partial η²" | analysis output | explained in `EXPERIMENT_PLAN.md`, not in the tool's output |
| "WNS", "Fmax" | sweep schema | glossary only |
| "vectorless estimate" | `build.tcl` | explained inline, but jargon-dense |
| "arithmetic intensity" | `mac_array.v` | explained thoroughly at that point |

**Recommendation:** on first use in `README.md`, gloss PCPI inline ("the
coprocessor port — see glossary") rather than relying on the link.

### Sharpest remaining edge

`sw/link.ld` and `start.S` are genuinely hard for a beginner. They are heavily
commented (including *why* there is no `.data` copy loop, and why
`.option norelax` is required around the `gp` setup), but linker scripts are
intrinsically unfamiliar. Fortunately nobody needs to touch them.

---

## 7. Risk register

Blunt, ordered by expected damage over the next six months.

### R1 — The board arrives and the FPGA core rail cannot be isolated · HIGH

Energy measurement requires cutting a shunt to get at the FPGA core supply. If
that proves impractical, whole-board current measurement will be dominated by
regulators, the FTDI bridge and LEDs, and **every configuration will look
identical**.

*Early warning:* the first PPK2 capture shows large current with no difference
between idle and active.
*Mitigation:* study the Arty A7 schematic **before** ordering. Identify the
shunt. If isolation is impossible, fall back to a relative differential
protocol (same board, same setup, only the bitstream changes) and state the
limitation explicitly.

### R2 — Nobody on the team can debug a waveform · HIGH

Four beginners, no prior Verilog. The first genuinely subtle timing bug will
cost days. The one already found (D008) was only visible by tracing PCPI
signals across cycles.

*Early warning:* someone says "it just doesn't work" without a waveform open.
*Mitigation:* everyone works through `VERILOG_PRIMER.md` §11 and deliberately
breaks something to see what a failure looks like — **before** it happens for
real. Practise on `tb_dot4` with an injected fault, as in §1 above.

### R3 — Vivado eats weeks · HIGH

~50 GB install, slow, cryptic errors, and macOS is unsupported. The sweep is
64 configurations × 5 replicates; if each takes 10 minutes that is 53 hours of
compute.

*Early warning:* the first synthesis run does not complete within a day of
starting.
*Mitigation:* install it **now**, on Linux, before it is on the critical path.
Run one configuration end to end early. Reduce replicates to 3 if needed and
say so. Consider synthesising a representative subset rather than all 64.

### R4 — Real data is much harder than the model · HIGH

MIMII is ~10 GB with licence acceptance; CWRU is MATLAB files. Preprocessing
must match *exactly* between training and deployment — and `data.py` already
warns that librosa and the numpy fallback are **not** bit-identical.

*Early warning:* accuracy is excellent on synthetic data and poor on real.
*Mitigation:* download one machine type early. Pin the spectrogram backend and
record it in metadata (already implemented). Never mix backends within an
experiment.

### R5 — The accelerated variants diverge and nobody notices · HIGH

`nn_dot4.c` and `nn_array.c` are not covered by full-model parity (Gap G1). A
divergence there would look like an accuracy difference between variants —
which is exactly what the project is trying to measure.

*Early warning:* the array variant's accuracy differs from the baseline's at
all. It should be **identical**, not similar.
*Mitigation:* close G1. Until then, treat any accuracy difference between
variants as a bug, not a result.

### R6 — Timing closure fails at 8×8 · MEDIUM

Broadcast fanout grows with width; Fmax will fall.

*Early warning:* negative WNS in the first wide synthesis.
*Mitigation:* `PIPELINE=1` in `dot4_pcpi.v` is already the escape hatch. Or
lower the clock and report it. **Never report a cycle count from a design that
fails timing as if it were achievable.**

### R7 — The headline result is "the accelerator barely matters" · MEDIUM

The energy model already suggests sleep leakage dominates. An Artix-7 is not a
low-power sleep device.

*Early warning:* the sensitivity analysis keeps pointing at sleep current.
*Mitigation:* **this is a legitimate finding, not a failure.** Report it. The
system-level framing ("accelerating inference is not where the energy is at
this duty cycle") is more interesting than another speedup number, and it is
defensible in a way that a buried result is not.

### R8 — Scope creep into a second workload too late · MEDIUM

Workload B is fully specified but has no trained model.

*Early warning:* it is December and workload A is not finished.
*Mitigation:* see the scope-cut guidance below.

### R9 — Four people editing RTL without review · MEDIUM

Silent breakage is easy in Verilog.

*Mitigation:* CI already gates on parity and simulation. Enforce "no merge
with red CI". Require a second reader on any `rtl/` change.

### R10 — The lab notebook stops being updated · MEDIUM

The most commonly abandoned artifact, and judges ask what did not work.

*Mitigation:* one entry per working session, written *during* the work.
Failures are the valuable part.

### Scope-cut guidance — what to abandon first, in order

If it is **December** and the array is broken:

1. **Cut the array; keep `dot4`.** A working custom instruction with clean
   measurements is a complete, publishable project. RQ1, RQ2, RQ4 and RQ5 are
   all still answerable. This is the single most important cut.
2. **Cut workload B.** Report RQ5 as future work. One workload, well
   characterized, beats two done badly.
3. **Cut int4.** Report RQ4 as future work; int8 alone is defensible.
4. **Cut the ESP32-S3 baseline.** Regrettable — it is the comparison judges
   want — but it is additive rather than load-bearing.
5. **Reduce the sweep to array width only**, at fixed buffer depth.

**Never cut, in any circumstance:**
- bit-exactness (`make test`) — without it no number means anything;
- direct energy measurement — replacing it with an estimate removes the
  project's central methodological claim;
- the lab notebook.

---

## 8. Gap list

Ordered by urgency. Effort estimates assume one student who has read the docs.

| # | Gap | Urgency | Effort | Notes |
|---|---|---|---|---|
| **G1** | `nn_dot4.c` / `nn_array.c` not covered by full-model parity | **Highest** | 1–2 days | Needs host stubs for the intrinsics, or a RISC-V simulator in the loop. Until then, a divergence in the accelerated path is invisible. |
| **G2** | Firmware never cross-compiled | High | 0.5 day | Install toolchain, `make -C sw`, run `tb_soc` with real firmware. Everything is written and the host build is clean under `-Werror`. |
| **G3** | RQ1 not answered | High | 0.5 day after G2 | The baseline MAC fraction sets the Amdahl ceiling and bounds every claim. Do it early — it may change the plan. |
| **G4** | Vivado flow never executed | High | 1–3 days | Includes install. **Verify `-generic` propagation on the first run** by diffing two configs' `summary.txt`. |
| **G5** | No real training run | High | 2–4 days | Blocked on datasets. Pipeline is verified with synthetic weights. |
| **G6** | Dataset URLs unverified | High | 0.5 day | `TODO_BLOCKED` TB001. |
| **G7** | No energy measurements | High | Blocked on PPK2 | All scripts written, never run. Treat first use as bring-up. |
| ~~G8~~ | ~~Verilator lint never run~~ | **CLOSED** | done | Verilator 5.050 now runs clean across all 11 modules. Found and fixed 3 real issues -- see below. |
| **G9** | Activation scales never calibrated | Medium | 1 day | Currently defaults. `quantize.py` warns. Accuracy is not meaningful until this is done. |
| **G10** | ESP32-S3 project is a README only | Medium | 2–3 days | Skeleton and fairness requirements documented. |
| **G11** | Workload B never trained or exported | Medium | 1–2 days | Config frozen; `deployed_model` sized to fit. |
| **G12** | Weight-stationary dataflow not implemented | Low | 2–3 days | `WEIGHT_REG` hook exists in `mac_unit.v`. A good follow-up study. |
| **G13** | No `requirements.lock` | Low | 1 hour | |
| **G14** | Non-square arrays not swept | Low | 1 hour | RTL supports and is tested at 2×8; excluded to keep the factorial at 64 cells. |
| **G15** | `int4` buffer packing is capacity-modelled, not bit-packed | Low | 1–2 days | Bandwidth advantage is modelled in `MEM_BEATS`; storage is not physically packed two-per-byte. Note when reporting int4 area. |

---

## Addendum: Verilator lint results (G8, closed)

Run for the first time with Verilator 5.050. It found three genuine issues
that Icarus tolerates, all now fixed. Recorded because they are a good
illustration of *why* a second, stricter tool is worth the install.

1. **A documentation comment had become a compiler directive.**
   `accel_pkg.vh` contained the line `//     verilator -GARRAY_W=4 ...` as
   part of a comment explaining how to override parameters. Verilator treats
   any comment whose first word is `verilator` as a pragma -- including `//`
   line comments -- so this parsed as a malformed directive and hard-errored
   every lint run. Fixed by prefixing with `$ `. Entirely self-inflicted, and
   completely invisible to Icarus.

2. **Implicit width extension in the `dot4` adder tree.** The four 16-bit
   products were summed into a 32-bit result, relying on Verilog's
   context-determined width rules to widen them. This was *not* a bug -- the
   `4 x (-128 x -128) = 65536` test case needs 17 bits and passes -- but it
   depended on one of Verilog's genuinely surprising rules. Now uses the
   explicit `{{16{p[15]}}, p}` sign-extension idiom, matching the style used
   elsewhere. The risk it removes is a future refactor introducing a 16-bit
   intermediate and silently truncating, with data-dependent symptoms.

3. **Mixed-width saturation comparison in `requantize.v`.** The 32-bit
   `QMIN`/`QMAX` limits were compared against the 64-bit intermediate.
   Verilog sign-extends correctly here, but this is the one module where a
   silent width error would corrupt every number the project produces, so the
   widths are now explicit on both sides.

Also tied off PicoRV32's unused `trace_valid`/`trace_data` ports explicitly,
so a genuinely missing connection is not lost in PINMISSING noise.

Two warning classes are waived, both in `third_party/picorv32` and neither
ours to fix: `GENUNNAMED` (a naming-style rule from IEEE 1800-2023) and
`BLKSEQ` (blocking assignments in clocked blocks, deliberate in that
codebase). **Our RTL must never trip BLKSEQ** -- if it ever does, remove the
waiver and fix it, because that is a real bug.

Bit-exactness was re-verified after all three fixes: 10/10 parity stages and
871 RTL checks still pass.

---

## Bottom line

The **foundation is solid and genuinely verified**: bit-exactness holds across
three independent implementations including a whole-model check, the SoC boots
and runs custom instructions on a real core, and the analysis pipeline has been
validated against known ground truth. Two real bugs were found and fixed, and
the test infrastructure demonstrably catches injected faults in both C and
Verilog.

The **largest technical risk is G1** — the accelerated software variants are
not yet parity-checked end to end, and that is precisely where an undetected
divergence would masquerade as a result.

The **largest project risk is R3/R4** — Vivado and real data are the two things
that reliably consume more time than teams expect, and neither is on the
critical path yet. Start both now.

The **most likely scientific outcome to prepare for** is R7: that inference is
a small share of node energy and the accelerator moves deployment lifetime
very little. That is a good result, honestly obtained, and the repository is
already built to report it rather than hide it.
