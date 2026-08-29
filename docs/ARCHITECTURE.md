# Architecture

Full technical design. Read [GLOSSARY.md](GLOSSARY.md) first if any term is
unfamiliar, and [VERILOG_PRIMER.md](VERILOG_PRIMER.md) if you have not written
hardware before.

---

## 1. The argument in one page

A CPU is built for branching, unpredictable code. Neural network inference is
the opposite: one arithmetic pattern — multiply-accumulate — repeated at
enormous volume with almost no branching.

Concretely, here is the inner loop of a quantized convolution in plain C
(`sw/src/nn_baseline.c`):

```c
acc += (int32_t)x_ic[iy * in_w + ix] * (int32_t)w_ic[ky * kw + kx];
```

On rv32i that becomes roughly: two loads, two sign-extensions, a multiply
(rv32i has **no** hardware multiply, so this is a called routine or a
shift-add sequence), an add, two pointer increments, a compare and a branch.
That is 8–10 instructions to perform **one** useful multiply-add.

The overhead is *structural*, not a compiler deficiency. The instruction set
has no way to express "do four of these at once", so it cannot be optimised
away — only architected away. This project measures how much of it can be
recovered, and at what silicon cost.

Three levels of intervention, each a separate build:

| Variant | Mechanism | Per-MAC cost |
|---|---|---|
| `baseline` | plain rv32i | ~8–10 instructions per MAC |
| `dot4` | custom instruction, 4 lanes | ~6 instructions per 4 MACs |
| `array` | memory-mapped MAC grid | `ARRAY_H × ARRAY_W` MACs/cycle |

---

## 2. System block diagram

```
                    +-------------------------------------------+
                    |               soc_top.v                   |
                    |                                           |
   clk  ----------->|  +------------+        +---------------+  |
   resetn --------->|  |  picorv32  |<--PCPI-| dot4_pcpi.v   |  |
                    |  |  (rv32i)   |        | DOT4 / DOT4A  |  |
                    |  +-----+------+        | ACCRD         |  |
                    |        |               +---------------+  |
                    |        | 32-bit memory bus                |
                    |   +----+----+----------+---------+-----+  |
                    |   |         |          |         |     |  |
                    | +-v--+  +---v---+  +---v----+  +-v---+ |  |
                    | |RAM |  | UART  |  | perf   |  |exit | |  |
                    | |64K |  | tx    |  | ctrs   |  |port | |  |
                    | +----+  +---+---+  +--------+  +-----+ |  |
                    |             |                          |  |
                    |        +----v-------------------+      |  |
                    |        |     accel_top.v        |      |  |
                    |        |  (memory-mapped @4000) |      |  |
                    |        |                        |      |  |
                    |        |  weight_buffer         |      |  |
                    |        |  activation_buffer     |      |  |
                    |        |  accel_ctrl (FSM)      |      |  |
                    |        |  mac_array HxW         |      |  |
                    |        |  perf_counter          |      |  |
                    |        |  result FIFO           |      |  |
                    |        +------------------------+      |  |
                    +-------------------------------------------+
                                       |
                                  uart_tx_pin --> host terminal
```

### Address map

| Range | Device |
|---|---|
| `0x0000_0000` – `MEM_BYTES-1` | RAM (program + data + stack) |
| `0x1000_0000` | UART data (write a byte to send) |
| `0x1000_0004` | UART status (bit 0 = busy) |
| `0x2000_0000` | free-running cycle counter |
| `0x2000_0004` | retired-instruction counter |
| `0x3000_0000` | simulation exit port (write → `$finish`) |
| `0x4000_0000` – `+0xFF` | accelerator registers |

Unmapped accesses are **answered and loudly warned about** rather than left to
hang. A bus transaction that nobody completes stalls the CPU forever with no
error message — the simulation simply goes quiet. That failure mode is the
single most confusing thing that can happen in a small SoC, so `soc_top.v`
has a catch-all that prints the offending address. **Do not delete it.**

---

## 3. Quantization and the requantization pipeline

### Why integer-only

The core has no FPU. A float multiply would be emulated in software at a cost
of dozens of instructions — dominating the very loop we are measuring. More
fundamentally, float results cannot be reproduced bit-exactly by the FPGA, so
the parity harness would be impossible and hardware bugs would be
indistinguishable from numerical drift.

### The scheme

Symmetric, per-tensor, zero point = 0:

```
q = clamp(round(x / scale)),   scale = max|x| / QMAX
```

Symmetric matters because padding and ReLU outputs are *exactly* zero. With an
asymmetric scheme, integer zero would represent some small nonzero float, and
that error would propagate through every padded convolution.

### The accumulate → requantize path

A layer accumulates int8 × int8 products into an **int32** accumulator. Its
real-world value is `acc × s_in × s_w`. To express that at the output scale we
need to multiply by

```
real_multiplier = (s_in × s_w) / s_out       ∈ (0, 1)
```

which is a real number. We convert it **once, offline** into an integer pair:

```
real_multiplier = m × 2^e            with m ∈ [0.5, 1), e ≤ 0   (frexp)
M0    = round(m × 2^31)              so M0 ∈ [2^30, 2^31)
shift = 31 − e                       so shift ≥ 31
```

and then at runtime

```
acc × real_multiplier  ==  (acc × M0) >> shift
```

using only integer multiply and shift.

**Edge case that must not be missed:** if `m` rounds up to exactly 1.0 then
`M0 = 2^31`, which does not fit in a signed 32-bit integer. `quantize.py`
halves `M0` and decrements the exponent. Omitting this produces a *negative*
`M0` in C and Verilog and flips the sign of an entire layer.

### The rounding rule

See [DECISIONS.md D003](DECISIONS.md) for the full rationale. Summary:

```
prod  = acc × M0                       (64-bit — int32 overflows here!)
half  = 1 << (shift − 1)
prod ≥ 0:  out =  ( prod + half) >> shift
prod < 0:  out = −((−prod + half) >> shift)
out += zero_point, then saturate to [QMIN, QMAX]
```

Round half **away from zero**, symmetrically. The explicit negate on the
negative branch is the load-bearing part: an arithmetic right shift rounds
toward −∞, so the naive form would round +2.5 → +3 but −2.5 → −2, injecting a
DC bias into every layer.

Two failure modes worth naming, both silent:
- **int32 intermediate.** `acc` can be 2³¹ and `M0` is near 2³¹, so the
  product needs 62 bits. Computing it in 32 overflows on nearly every input.
- **Clamp before narrowing.** Narrowing to int32 first would wrap a huge
  accumulator around and then clamp the wrong value.

### Three implementations, one behaviour

| Language | File | Role |
|---|---|---|
| Python | `train/quant_ref.py` | the reference |
| C | `sw/src/quant.c` | runs on the RISC-V core |
| Verilog | `rtl/requantize.v` | runs on the FPGA |

`train/verify_parity.py` proves all three agree, and includes a self-test that
deliberately breaks the rounding rule and **requires** the harness to notice.

---

## 4. The custom instructions (Phase 03)

RISC-V reserves opcode `0b0001011` (custom-0) for vendor extensions. We use
the R-type layout:

```
 31      25 24   20 19   15 14  12 11   7 6      0
+----------+-------+-------+------+------+--------+
|  funct7  |  rs2  |  rs1  |funct3|  rd  | opcode |
+----------+-------+-------+------+------+--------+
```

| Instruction | funct7 | Behaviour |
|---|---|---|
| `DOT4`  | `0000000` | `rd = Σ int8(rs1[i]) × int8(rs2[i])`, i = 0..3 |
| `DOT4A` | `0000001` | `acc += ` that sum; `rd` not written |
| `ACCRD` | `0000010` | `rd = acc; acc = 0` (atomic read-and-clear) |

C code reaches these through `sw/include/accel.h` using `.insn r`, so nobody
hand-assembles anything:

```c
static inline int32_t dot4(uint32_t a, uint32_t b) {
    int32_t rd;
    __asm__ volatile (".insn r 0x0B, 0, 0, %0, %1, %2"
                      : "=r"(rd) : "r"(a), "r"(b));
    return rd;
}
```

### The PCPI handshake, and the one-shot rule

PicoRV32 asserts `pcpi_valid` with the instruction and operand *values*, then
waits. The coprocessor answers with `pcpi_ready` (and `pcpi_wr`/`pcpi_rd` if
it writes a register), or asserts `pcpi_wait` if it needs more cycles.

Two hard rules:

1. **If the instruction is not ours, assert neither `wait` nor `ready`**, so
   the core can trap it as illegal.
2. **Never assert `wait` without eventually asserting `ready`** — that hangs
   the core silently.

And one that cost us a real bug (D008): the core holds `valid` high until it
*sees* `ready`, which is at least one cycle after we decide to send it. So the
instruction body must be guarded by a `responded` flag to execute **exactly
once**. Without it, `DOT4A` accumulates twice per instruction and every dot
product comes out at 2×.

`PIPELINE=1` is the escape hatch for timing closure: it registers the products
so the four multiplies and the adder tree no longer sit in one clock period.

### Why `dot4` helps fully-connected layers far more than convolutions

`dot4` needs **four contiguous bytes**. A fully-connected reduction is a long
contiguous run — ideal. A 3×3 convolution window walks a row and then jumps to
the next; only `kw = 3` elements are contiguous, which is fewer than 4. So the
convolution layers in workload A get **no benefit at all** from `dot4`.

That is a finding, not a defect to hide. It is also precisely the motivation
for the MAC array, which reorganises the data (im2col) so long contiguous
reductions exist.

---

## 5. The MAC array (Phase 04)

### Dataflow

Output-stationary broadcast. Each cell owns one output element `C[i][j]` and
keeps its accumulator there for the whole reduction:

```
                b_in[0]  b_in[1]  b_in[2]      <- weights, one per column
                   |        |        |
    a_in[0] ----[cell00][cell01][cell02]
    a_in[1] ----[cell10][cell11][cell12]
```

Each cycle we broadcast one column of A and one row of B; after K cycles every
cell holds a finished dot product.

### Arithmetic intensity — the core of RQ3

Per cycle the array consumes `H + W` bytes and performs `H × W` MACs:

| Array | MACs/cycle | Bytes/cycle | Intensity |
|---|---|---|---|
| 1×1 | 1 | 2 | 0.5 |
| 2×2 | 4 | 4 | 1.0 |
| 4×4 | 16 | 8 | 2.0 |
| 8×8 | 64 | 16 | 4.0 |

A wider array is not merely more compute — it is intrinsically *more efficient
per byte moved*, because each fetched byte feeds a whole row or column.

But the bus delivers only 4 bytes/cycle. An 8×8 array wants 16, so without
local buffering it can issue a MAC step only every `ceil(16/4) = 4` cycles;
the other 3 are **stalls**. That ratio is `MEM_BEATS` in `rtl/accel_ctrl.v`,
and it is the mechanism by which buffer depth and array width **interact**:
`MEM_BEATS` grows with width *only when there is no buffer*.

This is why the experiment is a full factorial. A one-factor-at-a-time study
would report two independent main effects and miss the actual finding.

Measured in simulation (`sweep/run_sweep.py --dry-run --quick`, workload A):

| Config | Cycles | Stalls | Utilisation |
|---|---|---|---|
| 1×1, no buffer | 1037 | 0 | 0.99 |
| 1×1, buffer 1024 | 1037 | 0 | 0.99 |
| 8×8, no buffer | 1028 | 768 | 0.25 |
| 8×8, buffer 1024 | 260 | 0 | 0.99 |

Buffering changes *nothing* at width 1 and gives a ~4× improvement at width 8.
That is the interaction, visible directly in the cycle counts.

### Control FSM

`rtl/accel_ctrl.v` implements the tiling loops as an explicit state machine —
the nested `for` loops of the C version turned inside out, because hardware
has no program counter:

```
S_IDLE → S_CLEAR → S_STREAM (K times) → S_DRAIN → S_NEXT → ... → S_DONE
```

`S_CLEAR` wipes accumulators before each tile; skipping it makes tile 2
contain tile 1's sum. Degenerate dimensions (`M`, `N` or `K` = 0) route
straight to `S_DONE` rather than wrapping a counter 65536 times.

### Precision

`PRECISION` ∈ {8, 4} affects three things, all separately measurable:
- the clamp range in quantization (accuracy — RQ4);
- MAC operand width, so synthesis infers a smaller multiplier (area);
- operand bytes per step, so int4 halves bandwidth demand (a real advantage
  that is easy to overlook when thinking only about multiplier size).

---

## 6. Performance counters

`rtl/perf_counter.v` maintains four counters, readable from C:

| Counter | Meaning |
|---|---|
| `cycles_total` | everything between start and done |
| `cycles_active` | a MAC step actually issued |
| `cycles_stall` | array was ready but had no operands |
| `macs_done` | useful multiply-accumulates retired |

`cycles_active` and `cycles_stall` deliberately do **not** sum to
`cycles_total`. A cycle can be neither — while the controller reloads a tile
descriptor — and the difference exposes control overhead as a third cost
centre rather than hiding it inside one of the other two.

From these: `utilisation = macs_done / (H·W·cycles_total)` and
`stall_fraction = cycles_stall / cycles_total`. A configuration with high
stall fraction is memory-bound, and adding array width will **not** help it —
a falsifiable prediction.

---

## 7. Parameterization

Every knob is a compile-time parameter, overridable per build:

```bash
iverilog -PARRAY_W=8 ...                 # Icarus
verilator -GARRAY_W=8 ...                # Verilator
synth_design -generic ARRAY_W=8 ...      # Vivado
```

| Parameter | Levels | Affects |
|---|---|---|
| `ARRAY_W`, `ARRAY_H` | 1, 2, 4, 8 | MACs/cycle, area, Fmax |
| `WBUF_DEPTH`, `ABUF_DEPTH` | 0, 64, 256, 1024 | stalls, BRAM |
| `PRECISION` | 8, 4 | accuracy, area, bandwidth |
| `ENABLE_DOT4`, `ENABLE_ACCEL` | 0, 1 | which variant is built |

`ENABLE_*` build the baseline with the extra hardware genuinely **absent**
rather than merely unused, so its area and leakage do not contaminate the
baseline measurement.

Verified propagation: `sim/tb_mac_array.v` passes at 1×1, 2×2, 8×8 and the
non-square 2×8, with the geometry reported at elaboration so a silently
defaulted parameter is visible in the log.

---

## 8. Memory budget

| Item | Size |
|---|---|
| RAM (`MEM_WORDS = 16384`) | 64 KB |
| Workload A weights (int8) | ~5.3 KB |
| Workload B deployed weights | ~8.8 KB |
| Stack | 4 KB |
| Activations (ping-pong ×2) | ~16 KB |

`sw/link.ld` contains an `ASSERT` that **fails the build** if code + data +
stack exceed RAM, rather than letting it manifest as runtime corruption.
`MEM_WORDS` in `rtl/soc_top.v` and `LENGTH` in `link.ld` must be changed
together.
