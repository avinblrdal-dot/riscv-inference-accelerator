# ===========================================================================
# Top-level Makefile
#
#   make setup     one-time environment setup
#   make test      EVERYTHING that can be verified without hardware
#   make sim       RTL simulation only
#   make parity    bit-exactness only  (the most important target)
#   make weights   generate a model + C header + golden vectors
#   make sweep     design-space exploration (simulation only)
#   make analysis  ANOVA, Pareto and figures
#   make clean
#
# `make test` is the gate. If it does not pass, nothing measured by this
# project is trustworthy.
# ===========================================================================

PYTHON  ?= python3
CONFIG  ?= train/config/workload_a.yaml
RUNDIR  ?= train/runs/workload_a
# Where `make weights` writes the C header and golden vectors. Override both
# together with CONFIG/RUNDIR when building the other workload, e.g.:
#   make weights CONFIG=train/config/workload_b.yaml RUNDIR=train/runs/workload_b \
#                MODELS_DIR=sw/models_b GOLDEN_DIR=sim/golden_b
# Leaving these at the defaults while pointing CONFIG at workload_b would
# silently overwrite workload_a's header -- both are regenerable build
# products, but the two workloads need to coexist so RQ5 can compare them.
MODELS_DIR ?= sw/models
GOLDEN_DIR ?= sim/golden

.PHONY: all help setup test parity sim sim-verbose weights sweep analysis \
        lint clean distclean check-tools submodules

all: help

help:
	@echo "riscv-inference-accelerator"
	@echo ""
	@echo "  make setup      create .venv and install Python dependencies"
	@echo "  make test       run every check that needs no hardware  <-- START HERE"
	@echo "  make parity     bit-exactness: Python == C == Verilog"
	@echo "  make sim        RTL testbenches under Icarus Verilog"
	@echo "  make weights    generate model + C header + golden vectors"
	@echo "  make sweep      design-space sweep (simulation only)"
	@echo "  make analysis   ANOVA + Pareto + figures"
	@echo "  make lint       Verilator lint (optional, stricter)"
	@echo "  make clean      remove build products"
	@echo ""
	@echo "  See docs/GETTING_STARTED.md if any tool is missing."

# ---------------------------------------------------------------------------
check-tools:
	@echo "Tool availability:"
	@command -v $(PYTHON)  >/dev/null && echo "  OK   python3" || echo "  MISS python3  (required)"
	@command -v iverilog   >/dev/null && echo "  OK   iverilog" || echo "  MISS iverilog (needed for 'make sim')"
	@command -v verilator  >/dev/null && echo "  OK   verilator" || echo "  --   verilator (optional lint)"
	@command -v vivado     >/dev/null && echo "  OK   vivado" || echo "  --   vivado (only for synthesis)"
	@$(PYTHON) -c "import numpy" 2>/dev/null && echo "  OK   numpy" || echo "  MISS numpy (required)"
	@$(PYTHON) -c "import torch" 2>/dev/null && echo "  OK   torch" || echo "  --   torch (only for training)"

submodules:
	@if [ ! -f third_party/picorv32/picorv32.v ]; then \
	  echo "Fetching PicoRV32 submodule..."; \
	  git submodule update --init --recursive; \
	fi

setup:
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r train/requirements.txt || \
	  (echo ""; \
	   echo "Some optional packages failed. The core harness needs only numpy:"; \
	   echo "  ./.venv/bin/pip install numpy"; \
	   echo "See docs/GETTING_STARTED.md.")
	@echo ""
	@echo "Activate with:  source .venv/bin/activate"

# ---------------------------------------------------------------------------
# The gate. Order matters: parity first, because a parity failure makes every
# later number meaningless.
# ---------------------------------------------------------------------------
test: submodules parity sim
	@echo ""
	@echo "======================================================================"
	@echo " ALL HARDWARE-FREE CHECKS PASSED"
	@echo "======================================================================"
	@echo " Verified: bit-exactness (Python == C == Verilog), RTL unit tests,"
	@echo " and a full SoC boot with the custom instructions executing."
	@echo ""
	@echo " NOT verified here (needs hardware -- see docs/DECISIONS.md):"
	@echo "   energy, FPGA area/timing, real datasets, cross-compiled firmware."

parity:
	@echo "--- bit-exactness ---"
	$(PYTHON) train/verify_parity.py

sim: submodules
	@echo "--- RTL simulation ---"
	./sim/run_icarus.sh

sim-verbose: submodules
	VCD=1 ./sim/run_icarus.sh

lint: submodules
	./sim/run_verilator.sh lint

# ---------------------------------------------------------------------------
# Model artifacts. --synthetic needs no PyTorch and no dataset, so the whole
# export -> compile -> simulate path is exercisable from a clean clone.
# ---------------------------------------------------------------------------
weights:
	@echo "--- freezing config ---"
	$(PYTHON) train/freeze.py --config $(CONFIG)
	@echo "--- quantizing (synthetic weights) ---"
	$(PYTHON) train/quantize.py --config $(CONFIG) --synthetic \
	    --out $(RUNDIR)/quantized.npz
	@echo "--- exporting header + golden vectors ---"
	$(PYTHON) train/export_weights.py --config $(CONFIG) \
	    --quantized $(RUNDIR)/quantized.npz \
	    --header-out $(MODELS_DIR)/model_weights.h \
	    --golden-dir $(GOLDEN_DIR)
	@echo ""
	@echo "NOTE: these weights are SYNTHETIC. They prove the pipeline works;"
	@echo "      they are not a trained model and produce no real accuracy."

firmware:
	$(MAKE) -C sw

# ---------------------------------------------------------------------------
sweep:
	$(PYTHON) sweep/run_sweep.py --dry-run

sweep-quick:
	$(PYTHON) sweep/run_sweep.py --dry-run --quick

sweep-full:
	@echo "Full sweep WITH Vivado synthesis. This takes hours."
	$(PYTHON) sweep/run_sweep.py

analysis:
	@if [ ! -f sweep/results/sweep_results.csv ]; then \
	  echo "No sweep results yet -- generating synthetic data so the analysis"; \
	  echo "pipeline can be exercised. These are NOT results."; \
	  $(PYTHON) analysis/load_results.py --synthetic; \
	  CSV=sweep/results/synthetic_results.csv; \
	else CSV=sweep/results/sweep_results.csv; fi; \
	$(PYTHON) analysis/anova.py --csv $$CSV; \
	$(PYTHON) analysis/pareto.py --csv $$CSV --min-accuracy 0.90; \
	$(PYTHON) analysis/plots.py --csv $$CSV

validate-analysis:
	$(PYTHON) analysis/anova.py --validate

# ---------------------------------------------------------------------------
clean:
	rm -rf sim/build sw/build analysis/figures/*.png
	rm -f sw/models/model_weights.h
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.vcd' -delete 2>/dev/null || true

distclean: clean
	rm -rf .venv train/runs data/cache sweep/results/*.csv sim/golden/*.npz \
	       sim/golden/*.json
