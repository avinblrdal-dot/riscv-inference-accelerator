# Getting started

Zero to running, for someone who has never done hardware. Follow the section
for your OS, then run the verification command after each install so you know
it worked before moving on.

**You do not need everything.** The tiers below are ordered by how much they
unlock:

| Tier | What you install | What you can do |
|---|---|---|
| 0 | Python 3.9+ and git | run the bit-exactness harness, quantize, export, analyse |
| 1 | + Icarus Verilog | run every RTL test and simulate the whole SoC |
| 2 | + RISC-V GCC | compile the real C firmware |
| 3 | + Vivado | synthesise, get area/timing numbers, build a bitstream |
| 4 | + Arty A7-100T and PPK2 | measure energy on real hardware |

**Start at tier 1.** It is a single package and it unlocks almost everything
interesting.

---

## macOS

### Tier 0 — Python

macOS ships Python 3. Check it:

```bash
python3 --version
```

Anything 3.9 or newer is fine. Then set up a virtual environment:

```bash
cd riscv-inference-accelerator
python3 -m venv .venv
source .venv/bin/activate
pip install -r train/requirements.txt
```

If the heavy packages (torch, librosa) fail, that is survivable — the core
harness needs only numpy:

```bash
pip install numpy
```

Verify:

```bash
python3 train/verify_parity.py --stage c
```

You should see `PARITY OK`.

### Tier 1 — Icarus Verilog

```bash
brew install icarus-verilog
```

No Homebrew? Install it from [brew.sh](https://brew.sh) first.

Verify:

```bash
iverilog -V | head -1        # expect "Icarus Verilog version 12.0" or newer
./sim/run_icarus.sh
```

You should see `ALL SIMULATION TESTS PASSED`.

Optional but recommended — Verilator (stricter linting) and GTKWave (waveforms):

```bash
brew install verilator gtkwave
./sim/run_verilator.sh lint
```

> **If `brew` is unavailable** (a locked-down school machine, for example),
> Icarus can be built from source with only `make`, `m4`, `perl` and `gperf`,
> which Xcode Command Line Tools already provide. You will need to build
> `autoconf` and `bison ≥ 3` first, since macOS ships a bison too old for
> Icarus's grammar. This is exactly how the reference environment for this
> repository was set up, so it is known to work.

### Tier 2 — RISC-V toolchain

```bash
brew tap riscv-software-src/riscv
brew install riscv-tools
```

Verify:

```bash
riscv64-unknown-elf-gcc --version
make -C sw
```

### Tier 3 — Vivado

Download **AMD Vivado ML Standard** (free, no licence needed for Artix-7)
from AMD's site. It is a very large download (~50 GB) and macOS is **not
officially supported** — most teams run it on Linux or in a VM. See the Ubuntu
section.

---

## Ubuntu / Debian

### Tier 0 — Python

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv git
cd riscv-inference-accelerator
python3 -m venv .venv && source .venv/bin/activate
pip install -r train/requirements.txt
python3 train/verify_parity.py --stage c
```

### Tier 1 — Icarus Verilog

```bash
sudo apt-get install iverilog gtkwave verilator
iverilog -V | head -1
./sim/run_icarus.sh
```

### Tier 2 — RISC-V toolchain

```bash
sudo apt-get install gcc-riscv64-unknown-elf
riscv64-unknown-elf-gcc --version
make -C sw
```

If your distribution lacks that package, use the prebuilt toolchain from
[xPack](https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases)
and add its `bin/` to `PATH`. The `sw/Makefile` auto-detects several common
prefixes, including `riscv-none-elf-`.

### Tier 3 — Vivado

1. Download **AMD Vivado ML Standard** from AMD (free account required).
2. Run the installer; select **Artix-7** device support (you can deselect
   everything else and save a lot of disk).
3. Install cable drivers:
   ```bash
   cd <vivado_install>/data/xicom/cable_drivers/lin64/install_script/install_drivers
   sudo ./install_drivers
   ```
4. Add Vivado to your shell:
   ```bash
   source /tools/Xilinx/Vivado/2023.2/settings64.sh
   ```

Verify:

```bash
vivado -version
```

---

## Windows (via WSL2)

Use WSL2. Native Windows builds of these tools exist but the scripts here
assume a POSIX shell.

```powershell
wsl --install -d Ubuntu
```

Then open Ubuntu and follow the **Ubuntu** section above for tiers 0–2.

**For tier 3, install Vivado on Windows natively**, not inside WSL — USB
passthrough for the JTAG programmer is far more reliable that way. Keep the
repository on the Windows filesystem so both sides can reach it:

```bash
cd /mnt/c/Users/<you>/riscv-inference-accelerator
```

> Note: builds are noticeably slower on `/mnt/c` than on the WSL filesystem.
> If simulation feels sluggish, clone a second copy inside WSL for simulation
> work and keep the Windows copy for Vivado.

---

## First run: prove everything works

From the repository root, in order:

```bash
# 1. Bit-exactness: Python == C == Verilog
python3 train/verify_parity.py
```
Expect `PARITY OK`. **If this fails, stop and fix it** — nothing else in the
project is trustworthy until it passes.

```bash
# 2. RTL tests
./sim/run_icarus.sh
```
Expect `ALL SIMULATION TESTS PASSED` (about 870 individual checks).

```bash
# 3. Boot a program on the simulated CPU
./sim/run_icarus.sh tb_soc boot
```
Expect `RVACCEL-BOOT-OK` printed over the simulated UART.

```bash
# 4. Run the custom instructions on the real core
./sim/run_icarus.sh tb_soc dot4
```
Expect `123` — one digit per passing subtest (DOT4, DOT4A, ACCRD).

```bash
# 5. Generate a model and check the whole pipeline
make weights
python3 train/verify_parity.py
```
The full-model stage should now report all layers matching.

```bash
# 6. Exercise the sweep and the analysis
python3 sweep/run_sweep.py --dry-run --quick
python3 analysis/anova.py --validate
python3 analysis/plots.py
```

Or just run everything:

```bash
make test
```

---

## What you can do without any hardware

Almost all of it. Specifically:

- prove bit-exactness across Python, C and Verilog;
- simulate the entire SoC, including booting a program and printing over UART;
- run every RTL unit test at every array geometry;
- train (with PyTorch) or generate a synthetic model, quantize, and export;
- run the full design-space sweep for **cycles and stalls**;
- validate the statistical analysis against known ground truth;
- produce all four publication figures.

What genuinely needs hardware: area and timing numbers (Vivado), energy
measurements (PPK2 + board), and real motor recordings.

---

## Common first-run problems

| Symptom | Cause | Fix |
|---|---|---|
| `iverilog: command not found` | tier 1 not installed | see above |
| `PicoRV32 submodule is missing` | submodule not fetched | `git submodule update --init --recursive` |
| `no RISC-V cross-compiler found` | tier 2 not installed | you probably do not need it yet — use `sim/gen_smoke_hex.py` |
| Simulation dies instantly at t≈0 | firmware hex missing | check `sim/build/firmware.hex` exists |
| `PARITY FAILED` | genuine mismatch | read the reported index and values; **do not proceed** |
| `ModuleNotFoundError: yaml` | PyYAML not installed | harmless — a built-in fallback parser takes over |

More in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
