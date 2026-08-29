# Verilog for people who know Python or C

You do not need to learn all of Verilog. You need about 15% of it, and you
need to unlearn one deeply held assumption. This document is that 15%, using
examples from this project's own RTL.

---

## 1. The one big idea: hardware is concurrent

In Python, statements run **in order**:

```python
a = 1
b = a + 1     # b is 2, because line 1 already happened
```

In Verilog, you are not writing instructions. You are **describing a circuit
that physically exists all at once**. Every line is active simultaneously,
forever.

```verilog
assign a = 1;
assign b = a + 1;   // this is a permanent equation, not a step
```

Think of a spreadsheet, not a script. `assign b = a + 1;` means "there is a
wire called `b`, and it is physically connected to an adder whose input is
`a`". If `a` changes, `b` changes — immediately, and always. Order of lines in
the file is irrelevant. You could swap those two lines and nothing changes.

**This is the mental shift.** Almost every beginner Verilog bug comes from
reading it as sequential code.

---

## 2. `wire` vs `reg`

| | `wire` | `reg` |
|---|---|---|
| What it is | a physical connection | a value that is *remembered* |
| Assigned by | `assign` (continuous) | inside `always` blocks |
| Analogy | a piece of copper | a variable that survives to next clock |

```verilog
wire [6:0] opcode = pcpi_insn[6:0];      // just naming some bits
reg  [31:0] acc;                          // holds a value between clocks
```

**The name `reg` is a historical misnomer.** It does not necessarily become a
hardware register. It means "assigned procedurally". Do not read anything more
into it.

`[6:0]` means 7 bits, numbered 6 down to 0. `[31:0]` is 32 bits. Bit 0 is the
least significant.

---

## 3. `always @(posedge clk)` — this is a flip-flop

```verilog
always @(posedge clk) begin
    if (!resetn)
        acc <= 0;
    else if (en)
        acc <= acc + prod;
end
```

Read it as: *"at the instant the clock rises, if reset is not asserted and
enable is high, `acc` takes on its old value plus `prod`."*

Everything inside updates **simultaneously**, at that instant. The statements
are not steps.

`@(posedge clk)` is the sensitivity list: "re-evaluate on the rising edge of
`clk`". `begin`/`end` is Verilog's `{`/`}`.

---

## 4. `<=` vs `=` — the classic trap

This is the single most common source of "my design works in simulation but
not in hardware" bugs.

**`<=` is non-blocking.** All right-hand sides are evaluated first, then all
left-hand sides are updated together:

```verilog
always @(posedge clk) begin
    a <= b;
    b <= a;        // this SWAPS a and b
end
```

Both reads happen before either write. This is genuinely how flip-flops
behave, and it is what you want.

**`=` is blocking.** It takes effect immediately, like C:

```verilog
always @(posedge clk) begin
    a = b;
    b = a;         // this does NOT swap -- both end up as b's old value
end
```

### The rule

> **Use `<=` in `always @(posedge clk)`. Use `=` in combinational blocks and
> in testbenches. Never mix them in the same block.**

Follow that rule and you will never think about this again.

---

## 5. Reset, and why it is upside down

PicoRV32 uses **active-low** reset, named `resetn`. The trailing `n` means
"negated": `resetn == 0` means *in reset*.

```verilog
if (!resetn) begin
    acc <= 0;          // held in reset
end else begin
    // normal operation
end
```

Getting this backwards holds your design in reset permanently, and the symptom
— nothing ever happens — looks identical to a dead clock. Check this first
when a module is inexplicably inert.

**Every register that matters must be reset.** An unreset register starts as
`x` (unknown) in simulation, and `x` propagates: `x + 1` is `x`, and
`if (x)` is unpredictable. A wave of red `x` values spreading through your
waveform almost always traces back to one missing reset.

---

## 6. Signed arithmetic — the bug that will bite you

Bit selects are **unsigned by default**. This is the most consequential gotcha
in this entire project.

```verilog
wire [7:0] a = 8'hFF;
wire [15:0] wrong = a * b;              // treats a as 255
wire signed [7:0] as = 8'hFF;
wire signed [15:0] right = as * bs;     // treats a as -1
```

Our weights and activations are **signed int8**. If a sign is lost, results
are wrong only for negative inputs — so the bug is data-dependent, passes half
your tests, and is miserable to find.

Verilog's rule: **an expression is signed only if *every* operand is signed.**
One stray unsigned operand silently makes the whole expression unsigned. That
is why `rtl/mac_unit.v` declares operands `signed` explicitly and
`rtl/dot4_pcpi.v` uses `$signed()` casts.

Sign-extension idiom, from `dot4_pcpi.v` (extending a 4-bit nibble to 8 bits):

```verilog
assign a0 = {{4{pcpi_rs1[3]}}, pcpi_rs1[3:0]};
```

`{a, b}` is concatenation. `{4{x}}` replicates `x` four times. So this
replicates the sign bit four times and appends the nibble.

---

## 7. Parameters

Parameters are compile-time constants — like C++ templates, not variables.

```verilog
module mac_array #(
    parameter ARRAY_H = 4,
    parameter ARRAY_W = 4
) ( ... );
```

Override them per build:

```bash
iverilog -Ptb_mac_array.H=8 ...
```

`localparam` is the same but cannot be overridden — use it for internal
constants so nobody accidentally changes them from outside.

---

## 8. `generate` — building repeated hardware

To create 64 MAC cells you do not write a loop that runs 64 times. You write a
`generate` block that **instantiates 64 physical copies**:

```verilog
genvar gi, gj;
generate
    for (gi = 0; gi < ARRAY_H; gi = gi + 1) begin : g_row
        for (gj = 0; gj < ARRAY_W; gj = gj + 1) begin : g_col
            mac_unit u_mac ( ... );
        end
    end
endgenerate
```

This is closer to a C macro expanding at compile time than to a `for` loop.
After elaboration there are literally 64 multipliers on the chip. The `: g_row`
labels are required and become part of the hierarchical name you will see in a
waveform viewer.

`generate` also does compile-time `if`, which is how `weight_buffer.v` builds
either a real RAM or the depth-0 pass-through, with no runtime cost:

```verilog
generate
    if (DEPTH == 0) begin : g_nobuf
        assign rd_data = bypass_data;
    end else begin : g_buf
        reg [WIDTH-1:0] mem [0:DEPTH-1];
        ...
    end
endgenerate
```

---

## 9. Flat vectors (an ugly but necessary workaround)

Verilog-2001 cannot pass a 2-D array through a module port. So we flatten:

```verilog
input wire [ARRAY_H*8-1:0] a_in;      // H bytes packed into one vector
wire signed [7:0] a_cell = a_in[gi*8 +: 8];
```

`a_in[gi*8 +: 8]` is an *indexed part-select*: "starting at bit `gi*8`, take 8
bits going up". It is ugly. SystemVerilog would let us write it properly, but
Verilog-2001 is what every tool supports reliably, including Vivado's most
dependable synthesis path.

**Off-by-one errors in these index expressions are common** and typically show
up as a transposed or shifted result matrix. That is exactly why
`sim/tb_mac_array.v` checks the *entire* output matrix against a reference
rather than spot-checking `[0][0]` — which is often still correct even when
the packing is wrong.

---

## 10. Testbenches

A testbench is Verilog that is **not hardware**. It never gets synthesised; it
exists only inside the simulator, to drive inputs and check outputs.

```verilog
`timescale 1ns / 1ps

module tb_example;
    reg clk = 0;
    reg resetn = 0;
    always #5 clk = ~clk;          // toggle every 5ns -> 100MHz

    dut_module dut (.clk(clk), .resetn(resetn), ...);

    initial begin
        @(posedge clk);
        resetn <= 1'b1;
        // ... drive stimulus, check results ...
        $finish;
    end
endmodule
```

Constructs that only exist in simulation: `initial`, `#5` (delay),
`$display`, `$finish`, `$readmemh`, `$random`.

### The golden rule of testbenches

> **Compute the expected answer a *different way* than the design does.**

If you check the design against itself you prove only self-consistency. In
`tb_mac_array.v` the reference uses ordinary 2-D arrays and behavioural
arithmetic — a genuinely independent path from the DUT's flat-vector packing.

### Always add a timeout

```verilog
initial begin
    #2_000_000;
    $display("TEST FAILED -- global timeout");
    $finish;
end
```

Without one, a deadlocked design makes the simulator hang forever with no
output. With one, you get a diagnosable failure. Every testbench in `sim/`
has one.

---

## 11. Reading waveforms

Generate a VCD:

```bash
VCD=1 ./sim/run_icarus.sh tb_dot4
gtkwave sim/build/tb_dot4.vcd
```

In GTKWave: pick a module in the hierarchy pane (top left), drag signals into
the wave window, then use `Ctrl+Alt+F` to zoom to fit.

**How to actually debug with it:** find the first moment the output is wrong,
then walk *backwards* through the inputs that produced it. Bugs are almost
always one clock edge earlier than where you first notice them.

Things to look for:
- **red `x`** — unknown value, usually a missing reset or an undriven wire;
- **a signal stuck high** — often a pulse that was never defaulted low (see
  the "default then override" pattern in `dot4_pcpi.v`);
- **`z`** — high-impedance, meaning nothing is driving that wire at all.

---

## 12. Style rules used throughout this project

1. **`<=` in clocked blocks, `=` in combinational.** Never mixed.
2. **Default pulses low, then override.** This guarantees single-cycle pulses:
   ```verilog
   always @(posedge clk) begin
       pcpi_ready <= 1'b0;          // default
       if (claimed) pcpi_ready <= 1'b1;   // override
   end
   ```
   A stuck-high `ready` makes the core accept garbage for later instructions.
3. **Every register is reset.**
4. **Signed means `signed` everywhere in the expression.**
5. **Every `case` has a `default`** — costs nothing, guarantees recovery to a
   known state.
6. **No magic numbers.** Named constants in `rtl/accel_pkg.vh`.
7. **Header comment on every module** explaining the concept before the code.

---

## 13. Where to go next

Read the RTL in this order — it is arranged roughly by increasing difficulty:

1. `rtl/uart_tx.v` — a small, self-contained state machine
2. `rtl/mac_unit.v` — one multiply-accumulate cell
3. `rtl/requantize.v` — pure combinational arithmetic
4. `rtl/mac_array.v` — `generate` blocks and flat vectors
5. `rtl/accel_ctrl.v` — a real FSM with nested loop counters
6. `rtl/dot4_pcpi.v` — protocol handshaking, including the one-shot guard
7. `rtl/soc_top.v` — putting it all together

Then run the tests and break something on purpose to see what the failure
looks like:

```bash
./sim/run_icarus.sh
```
