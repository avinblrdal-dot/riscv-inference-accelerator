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
LDFLAGS := $(ARCH_FLAGS) -nostdlib -T link.ld -Wl,--gc-sections

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
	  start.S $$(COMMON_SRCS) $$(VARIANT_SRCS) $(SRC)/main.c -o $$@
	@echo "  built $$@"

$(BUILD)/$(1).hex: $(BUILD)/$(1).elf
	$$(OBJCOPY) -O verilog --verilog-data-width=4 $$< $$@
	@echo "  hex   $$@"

$(BUILD)/$(1).lst: $(BUILD)/$(1).elf
	$$(OBJDUMP) -d -S $$< > $$@

$(1): $(BUILD)/$(1).hex $(BUILD)/$(1).lst
	@$$(OBJDUMP) -h $$(BUILD)/$(1).elf | awk '/\.text|\.rodata|\.bss/ {printf "  %-10s %s bytes\n", $$$$2, strtonum("0x" $$$$3)}'
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
