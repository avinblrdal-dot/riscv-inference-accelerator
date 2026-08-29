# ===========================================================================
# sw/Makefile -- Cross-compile the inference firmware for rv32i
# ===========================================================================
#
# Two build paths live here, and knowing which is which saves confusion:
#
#   1. CROSS build  -> a .hex the FPGA/simulator runs.  Needs a RISC-V GCC.
#   2. HOST build   -> a native binary for parity testing. Needs only cc.
#
# The host path is what train/verify_parity.py uses, which is why bit-exact
# parity can be proven on a laptop with no cross-compiler installed.
#
# Usage:
#   make            # all three firmware variants
#   make baseline   # just the unaccelerated one
#   make host       # native parity binaries
#   make clean
#
# The toolchain prefix varies by how you installed it. Override it:
#   make CROSS=riscv64-unknown-elf-
# ===========================================================================

# Try the common prefixes in order; the first that exists wins.
CROSS ?= $(firstword $(foreach p, \
           riscv32-unknown-elf- riscv64-unknown-elf- riscv-none-elf- \
           riscv64-elf- riscv32-elf-, \
           $(if $(shell command -v $(p)gcc 2>/dev/null),$(p))))

CC      := $(CROSS)gcc
OBJCOPY := $(CROSS)objcopy
OBJDUMP := $(CROSS)objdump
SIZE    := $(CROSS)size

BUILD := build
SRC   := src
INC   := include
MODELS := models

# -march=rv32i  : base integer ISA only. NO 'm' extension -- the whole point
#                 is that this core has no hardware multiply, so the baseline
#                 must not secretly use one.
# -mabi=ilp32   : matching ABI. A mismatch here produces link errors that read
#                 as "ABI is incompatible" and confuse everyone the first time.
# -O2           : the baseline must be optimised. Comparing an -O0 baseline
#                 against an -O2 accelerated build would be dishonest.
# -ffreestanding / -nostdlib : there is no operating system and no libc.
ARCH_FLAGS := -march=rv32i -mabi=ilp32
CFLAGS  := $(ARCH_FLAGS) -O2 -Wall -Wextra -Werror \
           -ffreestanding -nostdlib -fno-builtin \
           -fomit-frame-pointer -I$(INC) -I$(MODELS)
LDFLAGS := $(ARCH_FLAGS) -nostdlib -T link.ld -Wl,--gc-sections \
           -Wl,--no-warn-rwx-segments

# libgcc, and why it is REQUIRED rather than optional.
#
# rv32i has no multiply or divide instruction. So when the C says `a * b`,
# GCC cannot emit a single instruction -- it emits a CALL to a software
# routine in libgcc (__mulsi3, __divsi3, __muldi3, __udivdi3). `-nostdlib`
# excludes libgcc along with libc, which is why the first link failed with a
# wall of undefined references to those names.
#
# This is not an annoyance to work around; it IS the phenomenon this project
# measures. Every multiply in the inference loop is a function call: save
# registers, jump, shift-and-add loop, return. That is a large part of the
# 8-10 instructions of overhead per useful multiply-add that motivates the
# whole accelerator.
#
# We link libgcc explicitly, and ONLY libgcc -- still no libc, no startup
# files, no heap. Note it must come AFTER the object files on the command
# line, because the linker resolves symbols left to right.
LDLIBS := -lgcc

COMMON_SRCS := $(SRC)/quant.c $(SRC)/nn_baseline.c
VARIANT_SRCS := $(SRC)/nn_dot4.c $(SRC)/nn_array.c

.PHONY: all baseline dot4 array host clean check-toolchain

all: baseline dot4 array

check-toolchain:
	@if [ -z "$(CROSS)" ]; then \
	  echo "ERROR: no RISC-V cross-compiler found."; \
	  echo "  Looked for: riscv32-unknown-elf-gcc, riscv64-unknown-elf-gcc,"; \
	  echo "              riscv-none-elf-gcc, riscv64-elf-gcc"; \
	  echo ""; \
	  echo "  macOS:  brew tap riscv-software-src/riscv && brew install riscv-tools"; \
	  echo "  Ubuntu: sudo apt-get install gcc-riscv64-unknown-elf"; \
	  echo "  See docs/GETTING_STARTED.md."; \
	  echo ""; \
	  echo "  You do NOT need this to run the RTL smoke tests or the parity"; \
	  echo "  harness -- see sim/gen_smoke_hex.py and 'make host'."; \
	  exit 1; \
	fi

$(BUILD):
	@mkdir -p $(BUILD)

# --- one rule, three variants, selected by -D flags ------------------------
# Each variant is a SEPARATE binary so that the baseline measurement contains
# no accelerator code at all.
define BUILD_VARIANT
$(BUILD)/$(1).elf: check-toolchain | $(BUILD)
	$$(CC) $$(CFLAGS) -DVARIANT_BUILD_$(2) $$(LDFLAGS) \
	  start.S $$(COMMON_SRCS) $$(VARIANT_SRCS) $(SRC)/main.c \
	  $$(LDLIBS) -o $$@
	@echo "  built $$@"

$(BUILD)/$(1).hex: $(BUILD)/$(1).elf
	$$(OBJCOPY) -O verilog --verilog-data-width=4 $$< $$@
	@echo "  hex   $$@"

$(BUILD)/$(1).lst: $(BUILD)/$(1).elf
	$$(OBJDUMP) -d -S $$< > $$@

$(1): $(BUILD)/$(1).hex $(BUILD)/$(1).lst
	@# Use the toolchain's own `size` rather than parsing objdump with awk.
	@# strtonum() is a GNU awk extension that macOS's BSD awk does not have,
	@# so the awk version failed on a Mac while working fine on Linux.
	@$$(SIZE) -A -d $$(BUILD)/$(1).elf 2>/dev/null | \
	  awk '/^\.text|^\.rodata|^\.data|^\.bss/ {printf "  %-10s %8d bytes\n", $$$$1, $$$$2} \
	       /^Total/ {printf "  %-10s %8d bytes\n", "TOTAL", $$$$2}'
endef

$(eval $(call BUILD_VARIANT,baseline,BASELINE))
$(eval $(call BUILD_VARIANT,dot4,DOT4))
$(eval $(call BUILD_VARIANT,array,ARRAY))

# --- host build, for parity testing ---------------------------------------
# No cross-compiler needed. train/verify_parity.py builds its own harness the
# same way; this target just proves the sources compile cleanly for the host.
host: | $(BUILD)
	$(or $(CC_HOST),cc) -std=c99 -O1 -Wall -Wextra -Werror \
	  -I$(INC) -c $(SRC)/quant.c -o $(BUILD)/quant_host.o
	$(or $(CC_HOST),cc) -std=c99 -O1 -Wall -Wextra -Werror \
	  -I$(INC) -c $(SRC)/nn_baseline.c -o $(BUILD)/nn_baseline_host.o
	@echo "  host build OK (quant.c, nn_baseline.c)"

clean:
	rm -rf $(BUILD)
