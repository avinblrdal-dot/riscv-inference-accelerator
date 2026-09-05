#!/usr/bin/env python3
"""Prove that Python, C and Verilog compute IDENTICAL bits.

WHY THIS IS THE MOST IMPORTANT SCRIPT IN THE REPOSITORY
-------------------------------------------------------
Three independent implementations of the same quantized arithmetic exist:

    Python   train/quant_ref.py    the reference
    C        sw/src/quant.c        what runs on the RISC-V core
    Verilog  rtl/requantize.v      what runs on the FPGA

If they ever disagree, every downstream number becomes uninterpretable. A 2%
accuracy drop could mean "int4 is genuinely too coarse" (a finding) or "the
sign extension in the MAC array is broken" (a bug), and without this harness
there is no way to tell which. Worse, the bug interpretation is the one
nobody checks, because the result looks plausible.

So: this runs first, it runs in CI, and a failure blocks everything else.

WHAT IT CHECKS
--------------
  Stage 1  Python vs C     on requantize, saturate, argmax, and DOT4
  Stage 2  Python vs Verilog on requantize (via Icarus)
  Stage 3  Python vs Verilog on the DOT4 custom instruction
  Stage 4  self-test: deliberately inject an error and confirm the harness
           CATCHES it. A test suite that cannot fail is not a test suite.

Every stage degrades gracefully: if a toolchain is missing the stage is
SKIPPED with an explanation, never silently passed.

Usage:
    python3 train/verify_parity.py                # everything available
    python3 train/verify_parity.py --stage c      # just Python vs C
    python3 train/verify_parity.py --n 5000       # more vectors
    python3 train/verify_parity.py --seed 7       # different vectors
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import quant_ref as qr  # noqa: E402


# --------------------------------------------------------------------------
# Small reporting helpers -- a parity failure must say EXACTLY where.
# --------------------------------------------------------------------------

class Reporter:
    """Collects results so the run ends with one clear verdict."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.skipped: list[tuple[str, str]] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.passed.append(name)
        print(f"  PASS  {name}{(' -- ' + detail) if detail else ''}")

    def fail(self, name: str, detail: str) -> None:
        self.failed.append(name)
        print(f"  FAIL  {name}")
        for line in detail.splitlines():
            print(f"        {line}")

    def skip(self, name: str, why: str) -> None:
        self.skipped.append((name, why))
        print(f"  SKIP  {name} -- {why}")

    def verdict(self) -> int:
        print()
        print("=" * 70)
        print(f" parity: {len(self.passed)} passed, {len(self.failed)} failed, "
              f"{len(self.skipped)} skipped")
        print("=" * 70)
        if self.skipped:
            print(" Skipped stages (install the tool to enable):")
            for name, why in self.skipped:
                print(f"   - {name}: {why}")
        if self.failed:
            print()
            print(" BIT-EXACTNESS IS BROKEN. Do not trust any measurement")
            print(" until this is fixed. Failed stages:")
            for name in self.failed:
                print(f"   - {name}")
            print(" PARITY FAILED")
            return 1
        print(" PARITY OK")
        return 0


def first_mismatch(a, b):
    """Return (index, expected, actual) for the first differing element."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        return (-1, f"shape {a.shape}", f"shape {b.shape}")
    diff = np.nonzero(a != b)[0]
    if len(diff) == 0:
        return None
    i = int(diff[0])
    return (i, int(a[i]), int(b[i]))


# --------------------------------------------------------------------------
# Test vector generation -- deterministic, seeded, and edge-case heavy
# --------------------------------------------------------------------------

def make_requant_vectors(n: int, seed: int):
    """Random plus directed edge cases for requantize().

    The directed cases matter more than the random ones: exact halves are
    where a rounding-rule disagreement shows up, and they essentially never
    occur by chance.
    """
    rng = random.Random(seed)
    vecs = []

    for _ in range(n):
        acc = rng.randint(-(2 ** 26), 2 ** 26)
        m0 = rng.randint(2 ** 30, 2 ** 31 - 1)
        shift = rng.randint(31, 45)
        zp = rng.choice([0, 0, 0, -5, 7])
        prec = rng.choice([8, 8, 8, 4])
        vecs.append((acc, m0, shift, zp, prec))

    # Exact-half cases: acc * 2^30 >> 31 lands exactly on .5 for odd acc.
    half = 1 << 30
    for acc in (1, -1, 3, -3, 5, -5, 7, -7, 2, -2):
        vecs.append((acc, half, 31, 0, 8))

    # Saturation on both sides, at both precisions.
    for prec in (8, 4):
        vecs.append((2 ** 26, 2 ** 31 - 1, 31, 0, prec))
        vecs.append((-(2 ** 26), 2 ** 31 - 1, 31, 0, prec))

    # Zero, and the extremes of the accumulator range.
    vecs.append((0, half, 31, 0, 8))
    vecs.append((2 ** 31 - 1, half, 40, 0, 8))
    vecs.append((-(2 ** 31), half, 40, 0, 8))
    return vecs


def make_dot4_vectors(n: int, seed: int):
    rng = random.Random(seed + 1)
    vecs = [
        (0x00000000, 0x00000000),
        (0x80808080, 0x80808080),   # all -128: the overflow probe
        (0x7F7F7F7F, 0x7F7F7F7F),   # all +127
        (0x01020304, 0xFF01FF01),   # mixed sign
        (0x000000FF, 0x00000001),   # one negative lane only
    ]
    for _ in range(n):
        vecs.append((rng.getrandbits(32), rng.getrandbits(32)))
    return vecs


# --------------------------------------------------------------------------
# Stage 1: Python vs C
# --------------------------------------------------------------------------

C_HARNESS = r"""
/* Auto-generated by train/verify_parity.py -- host-side parity harness.
 * Compiled natively (not for RISC-V) so we can compare the SAME C source
 * that runs on the core against the Python reference, without needing a
 * cross-compiler or a simulator. */
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

/* Mirror of quant_ref.dot4(): the software model of the DOT4 instruction. */
static int32_t soft_dot4(uint32_t a, uint32_t b, int32_t precision)
{
    int32_t total = 0;
    for (int lane = 0; lane < 4; lane++) {
        int32_t av = (int32_t)((a >> (lane * 8)) & 0xFF);
        int32_t bv = (int32_t)((b >> (lane * 8)) & 0xFF);
        if (precision == 4) {
            av &= 0x0F; bv &= 0x0F;
            if (av & 0x8) av -= 16;
            if (bv & 0x8) bv -= 16;
        } else {
            if (av & 0x80) av -= 256;
            if (bv & 0x80) bv -= 256;
        }
        total += av * bv;
    }
    return total;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <mode>\n", argv[0]); return 2; }

    if (argv[1][0] == 'r') {                 /* requantize */
        long acc, m0, shift, zp, prec;
        while (scanf("%ld %ld %ld %ld %ld", &acc, &m0, &shift, &zp, &prec) == 5) {
            printf("%d\n", nn_requantize((int32_t)acc, (int32_t)m0,
                                         (int32_t)shift, (int32_t)zp,
                                         (int32_t)prec));
        }
    } else if (argv[1][0] == 'd') {          /* dot4 */
        unsigned long a, b; long prec;
        while (scanf("%lx %lx %ld", &a, &b, &prec) == 3) {
            printf("%d\n", soft_dot4((uint32_t)a, (uint32_t)b, (int32_t)prec));
        }
    } else if (argv[1][0] == 's') {          /* saturate */
        long v, prec;
        while (scanf("%ld %ld", &v, &prec) == 2) {
            printf("%d\n", nn_saturate((int32_t)v, (int32_t)prec));
        }
    } else { fprintf(stderr, "unknown mode\n"); return 2; }
    return 0;
}
"""


def build_c_harness(workdir: str, extra_cflags: list[str] | None = None) -> str | None:
    """Compile sw/src/quant.c plus the harness above into a host binary."""
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") \
        or shutil.which("clang")
    if cc is None:
        return None

    harness_c = os.path.join(workdir, "parity_harness.c")
    with open(harness_c, "w") as fh:
        fh.write(C_HARNESS)

    binary = os.path.join(workdir, "parity_harness")
    cmd = [
        cc, "-std=c99", "-O1", "-Wall", "-Wextra",
        "-I", os.path.join(ROOT, "sw", "include"),
        harness_c, os.path.join(ROOT, "sw", "src", "quant.c"),
        "-o", binary,
    ]
    if extra_cflags:
        cmd[1:1] = extra_cflags
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        return None
    return binary


def run_c(binary: str, mode: str, lines: list[str]) -> list[int]:
    res = subprocess.run([binary, mode], input="\n".join(lines) + "\n",
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"C harness failed: {res.stderr}")
    return [int(x) for x in res.stdout.split()]


def stage_python_vs_c(rep: Reporter, n: int, seed: int,
                      workdir: str, extra_cflags=None,
                      label_suffix: str = "") -> None:
    binary = build_c_harness(workdir, extra_cflags)
    if binary is None:
        rep.skip("Python vs C" + label_suffix,
                 "no host C compiler found (set CC, or install clang/gcc)")
        return

    # --- requantize ---
    vecs = make_requant_vectors(n, seed)
    expected = [int(qr.requantize(a, m, s, z, p)) for a, m, s, z, p in vecs]
    got = run_c(binary, "r", [f"{a} {m} {s} {z} {p}" for a, m, s, z, p in vecs])
    mm = first_mismatch(expected, got)
    if mm is None:
        rep.ok(f"requantize: Python == C{label_suffix}", f"{len(vecs)} vectors")
    else:
        i, e, g = mm
        a, m, s, z, p = vecs[i] if i >= 0 else (0,) * 5
        rep.fail(f"requantize: Python == C{label_suffix}",
                 f"first mismatch at index {i}\n"
                 f"  inputs: acc={a} m0={m} shift={s} zp={z} precision={p}\n"
                 f"  python expected: {e}\n"
                 f"  C returned:      {g}")

    # --- dot4 ---
    dvecs = make_dot4_vectors(n // 4, seed)
    dexp = [qr.dot4(a, b, 8) for a, b in dvecs]
    dgot = run_c(binary, "d", [f"{a:x} {b:x} 8" for a, b in dvecs])
    mm = first_mismatch(dexp, dgot)
    if mm is None:
        rep.ok(f"dot4: Python == C{label_suffix}", f"{len(dvecs)} vectors")
    else:
        i, e, g = mm
        a, b = dvecs[i] if i >= 0 else (0, 0)
        rep.fail(f"dot4: Python == C{label_suffix}",
                 f"first mismatch at index {i}\n"
                 f"  inputs: a=0x{a:08x} b=0x{b:08x}\n"
                 f"  python expected: {e}\n  C returned: {g}")

    # --- saturate ---
    rng = random.Random(seed + 2)
    svecs = [(rng.randint(-1000, 1000), rng.choice([8, 4])) for _ in range(200)]
    svecs += [(127, 8), (128, 8), (-128, 8), (-129, 8), (7, 4), (8, 4), (-8, 4), (-9, 4)]
    sexp = [int(np.clip(v, qr.qmin(p), qr.qmax(p))) for v, p in svecs]
    sgot = run_c(binary, "s", [f"{v} {p}" for v, p in svecs])
    mm = first_mismatch(sexp, sgot)
    if mm is None:
        rep.ok(f"saturate: Python == C{label_suffix}", f"{len(svecs)} vectors")
    else:
        i, e, g = mm
        rep.fail(f"saturate: Python == C{label_suffix}",
                 f"index {i}: input {svecs[i]}, expected {e}, got {g}")



# --------------------------------------------------------------------------
# Stage: layer kernels (conv2d / fc) -- Python vs C
# --------------------------------------------------------------------------

LAYER_HARNESS = r"""
/* Auto-generated by train/verify_parity.py.
 * Exercises the REAL layer kernels from sw/src/nn_baseline.c against the
 * Python reference. Requantize parity alone is not enough: the layer kernels
 * are where indexing, padding and channel-stride bugs live, and those are
 * invisible to a scalar test. */
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

static int8_t  g_in[64 * 64];
static int8_t  g_w[64 * 64];
static int32_t g_b[64];
static int8_t  g_out[64 * 64];
static int32_t g_scratch[64 * 64];

int main(int argc, char **argv)
{
    if (argc < 2) return 2;

    if (argv[1][0] == 'c') {              /* conv2d */
        long in_ch, in_h, in_w, out_ch, k, stride, pad, mult, shift, prec;
        if (scanf("%ld %ld %ld %ld %ld %ld %ld %ld %ld %ld",
                  &in_ch, &in_h, &in_w, &out_ch, &k, &stride, &pad,
                  &mult, &shift, &prec) != 10) return 3;

        long n_in = in_ch * in_h * in_w;
        for (long i = 0; i < n_in; i++)  { long v; if (scanf("%ld", &v)!=1) return 3; g_in[i] = (int8_t)v; }
        long n_w = out_ch * in_ch * k * k;
        for (long i = 0; i < n_w; i++)   { long v; if (scanf("%ld", &v)!=1) return 3; g_w[i]  = (int8_t)v; }
        for (long i = 0; i < out_ch; i++){ long v; if (scanf("%ld", &v)!=1) return 3; g_b[i]  = (int32_t)v; }

        nn_conv_t layer;
        layer.weights = g_w; layer.bias = g_b;
        layer.out_ch = (int32_t)out_ch; layer.in_ch = (int32_t)in_ch;
        layer.kh = (int32_t)k; layer.kw = (int32_t)k;
        layer.stride = (int32_t)stride; layer.pad = (int32_t)pad;
        layer.qp.multiplier = (int32_t)mult; layer.qp.shift = (int32_t)shift;
        layer.qp.zero_point = 0; layer.qp.precision = (int32_t)prec;

        int32_t oh, ow;
        nn_conv2d(g_in, (int32_t)in_h, (int32_t)in_w, &layer, g_out, &oh, &ow, g_scratch);
        for (int32_t i = 0; i < out_ch * oh * ow; i++) printf("%d\n", g_out[i]);

    } else if (argv[1][0] == 'f') {       /* fully connected */
        long in_dim, out_dim, mult, shift, prec;
        if (scanf("%ld %ld %ld %ld %ld", &in_dim, &out_dim, &mult, &shift, &prec) != 5) return 3;
        for (long i = 0; i < in_dim; i++)          { long v; if (scanf("%ld",&v)!=1) return 3; g_in[i]=(int8_t)v; }
        for (long i = 0; i < in_dim*out_dim; i++)  { long v; if (scanf("%ld",&v)!=1) return 3; g_w[i]=(int8_t)v; }
        for (long i = 0; i < out_dim; i++)         { long v; if (scanf("%ld",&v)!=1) return 3; g_b[i]=(int32_t)v; }

        nn_fc_t layer;
        layer.weights = g_w; layer.bias = g_b;
        layer.out_dim = (int32_t)out_dim; layer.in_dim = (int32_t)in_dim;
        layer.qp.multiplier = (int32_t)mult; layer.qp.shift = (int32_t)shift;
        layer.qp.zero_point = 0; layer.qp.precision = (int32_t)prec;

        nn_fc(g_in, &layer, g_out);
        for (int32_t i = 0; i < out_dim; i++) printf("%d\n", g_out[i]);

    } else if (argv[1][0] == 'p') {       /* maxpool */
        long ch, h, w, k;
        if (scanf("%ld %ld %ld %ld", &ch, &h, &w, &k) != 4) return 3;
        for (long i = 0; i < ch*h*w; i++) { long v; if (scanf("%ld",&v)!=1) return 3; g_in[i]=(int8_t)v; }
        int32_t oh, ow;
        nn_maxpool2d(g_in, (int32_t)ch, (int32_t)h, (int32_t)w, (int32_t)k, g_out, &oh, &ow);
        for (int32_t i = 0; i < ch*oh*ow; i++) printf("%d\n", g_out[i]);
    } else return 2;
    return 0;
}
"""


def build_layer_harness(workdir: str) -> str | None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") \
        or shutil.which("clang")
    if cc is None:
        return None
    src = os.path.join(workdir, "layer_harness.c")
    with open(src, "w") as fh:
        fh.write(LAYER_HARNESS)
    binary = os.path.join(workdir, "layer_harness")
    res = subprocess.run(
        [cc, "-std=c99", "-O1", "-Wall", "-Wextra",
         "-I", os.path.join(ROOT, "sw", "include"), src,
         os.path.join(ROOT, "sw", "src", "quant.c"),
         os.path.join(ROOT, "sw", "src", "nn_baseline.c"), "-o", binary],
        capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        return None
    return binary


def stage_layers(rep: Reporter, seed: int, workdir: str) -> None:
    """Check the real conv/fc/maxpool kernels, not just scalar requantize."""
    binary = build_layer_harness(workdir)
    if binary is None:
        rep.skip("layer kernels: Python vs C", "no host C compiler")
        return

    rng = np.random.default_rng(seed)

    # ---- conv2d, across several shapes including padding and stride ----
    conv_cases = [
        # (in_ch, in_h, in_w, out_ch, k, stride, pad)
        (1, 8, 8, 4, 3, 1, 1),    # the shape our model actually uses
        (1, 8, 8, 4, 3, 1, 0),    # no padding
        (3, 6, 6, 2, 3, 1, 1),    # multi-channel
        (2, 7, 7, 3, 3, 2, 1),    # stride 2, odd size -> exercises rounding
        (1, 5, 5, 1, 1, 1, 0),    # 1x1 kernel
        (4, 5, 5, 2, 2, 1, 0),    # even kernel
    ]
    conv_fail = 0
    for (ic, ih, iw, oc, k, st, pad) in conv_cases:
        x = rng.integers(-128, 128, size=(ic, ih, iw)).astype(np.int8)
        w = rng.integers(-128, 128, size=(oc, ic, k, k)).astype(np.int8)
        b = rng.integers(-10000, 10000, size=(oc,)).astype(np.int32)
        m0, shift = qr.quantize_multiplier(0.002)

        acc = qr.conv2d_int(x, w, b, stride=st, pad=pad)
        exp = qr.requantize(acc, m0, shift, 0, 8).ravel().tolist()

        payload = [f"{ic} {ih} {iw} {oc} {k} {st} {pad} {m0} {shift} 8"]
        payload += [" ".join(str(int(v)) for v in x.ravel())]
        payload += [" ".join(str(int(v)) for v in w.ravel())]
        payload += [" ".join(str(int(v)) for v in b.ravel())]
        got = run_c(binary, "c", payload)

        mm = first_mismatch(exp, got)
        if mm is not None:
            i, e, g = mm
            rep.fail("conv2d: Python == C",
                     f"shape in={ic}x{ih}x{iw} out_ch={oc} k={k} "
                     f"stride={st} pad={pad}\n"
                     f"  first mismatch at output index {i}\n"
                     f"  python expected: {e}\n  C returned: {g}")
            conv_fail += 1
            break
    if conv_fail == 0:
        rep.ok("conv2d: Python == C", f"{len(conv_cases)} shapes")

    # ---- fully connected ----
    fc_fail = 0
    for (in_dim, out_dim) in [(16, 4), (13, 4), (64, 8), (1, 3), (7, 1)]:
        x = rng.integers(-128, 128, size=(in_dim,)).astype(np.int8)
        w = rng.integers(-128, 128, size=(out_dim, in_dim)).astype(np.int8)
        b = rng.integers(-10000, 10000, size=(out_dim,)).astype(np.int32)
        m0, shift = qr.quantize_multiplier(0.005)

        acc = qr.linear_int(x, w, b)
        exp = qr.requantize(acc, m0, shift, 0, 8).ravel().tolist()

        payload = [f"{in_dim} {out_dim} {m0} {shift} 8"]
        payload += [" ".join(str(int(v)) for v in x.ravel())]
        payload += [" ".join(str(int(v)) for v in w.ravel())]
        payload += [" ".join(str(int(v)) for v in b.ravel())]
        got = run_c(binary, "f", payload)

        mm = first_mismatch(exp, got)
        if mm is not None:
            i, e, g = mm
            rep.fail("fc: Python == C",
                     f"in_dim={in_dim} out_dim={out_dim}\n"
                     f"  first mismatch at index {i}\n"
                     f"  python expected: {e}\n  C returned: {g}")
            fc_fail += 1
            break
    if fc_fail == 0:
        rep.ok("fc: Python == C", "5 shapes")

    # ---- maxpool ----
    pool_fail = 0
    for (ch, h, w_, k) in [(1, 8, 8, 2), (3, 6, 6, 2), (2, 9, 9, 3)]:
        x = rng.integers(-128, 128, size=(ch, h, w_)).astype(np.int8)
        exp = qr.maxpool2d_int(x, k=k).ravel().tolist()
        payload = [f"{ch} {h} {w_} {k}",
                   " ".join(str(int(v)) for v in x.ravel())]
        got = run_c(binary, "p", payload)
        mm = first_mismatch(exp, got)
        if mm is not None:
            i, e, g = mm
            rep.fail("maxpool: Python == C",
                     f"ch={ch} {h}x{w_} k={k}: index {i}, expected {e}, got {g}")
            pool_fail += 1
            break
    if pool_fail == 0:
        rep.ok("maxpool: Python == C", "3 shapes")



# --------------------------------------------------------------------------
# Stage: the WHOLE model -- Python vs C, layer by layer
# --------------------------------------------------------------------------

def discover_model_dirs() -> list[tuple[str, str, str]]:
    """Find every (label, header_dir, golden_dir) pair that has been built.

    A separate header/golden pair exists per workload AND per precision --
    "sw/models" + "sim/golden" for workload A at int8, "sw/models_b" +
    "sim/golden_b" for workload B, "sw/models_int4" + "sim/golden_int4" for
    the RQ4 precision sweep, and so on. Discovering them by directory name
    rather than hardcoding "sw/models" is what lets this check scale to a
    second workload without silently checking only one of them (or checking
    an arbitrary one, which is what picking npzs[0] used to do).
    """
    sw_dir = os.path.join(ROOT, "sw")
    pairs = []
    if not os.path.isdir(sw_dir):
        return pairs
    for name in sorted(os.listdir(sw_dir)):
        if name != "models" and not name.startswith("models_"):
            continue
        header_dir = os.path.join(sw_dir, name)
        if not os.path.exists(os.path.join(header_dir, "model_weights.h")):
            continue
        golden_name = "golden" + name[len("models"):]  # models_b -> golden_b
        golden_dir = os.path.join(ROOT, "sim", golden_name)
        pairs.append((name, header_dir, golden_dir))
    return pairs


def stage_full_model(rep: Reporter, workdir: str) -> None:
    """Run every exported model end to end in C and diff every intermediate.

    This is the strongest check in the suite. The scalar and layer stages
    prove the primitives agree; this proves they agree when COMPOSED, with the
    real generated weights, the real topology, and the real buffer
    ping-ponging in between. Composition is where shape and stride bugs live.

    Diffing every LAYER rather than just the final output matters: a bug in
    an early layer can be masked by the final layer (argmax, or an
    autoencoder's own averaging), so a matching final value is weak evidence
    while matching every intermediate is strong evidence.

    Runs once per discovered (workload, precision) header/golden pair -- see
    discover_model_dirs -- so adding workload B does not silently leave it
    unchecked. Skipped (not failed) when nothing has been generated yet,
    since headers and golden vectors are build products -- run `make weights`.
    """
    pairs = discover_model_dirs()
    if not pairs:
        rep.skip("full model: Python vs C",
                 "no sw/models*/model_weights.h found -- run 'make weights'")
        return

    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        rep.skip("full model: Python vs C", "no host C compiler")
        return

    for label, header_dir, golden_dir in pairs:
        check_name = f"full model: Python vs C [{label}]"

        npzs = ([f for f in os.listdir(golden_dir) if f.endswith(".npz")]
                if os.path.isdir(golden_dir) else [])
        if not npzs:
            rep.skip(check_name,
                     f"no golden vectors in {os.path.relpath(golden_dir, ROOT)} "
                     f"-- run 'make weights' for this workload/precision")
            continue
        if len(npzs) > 1:
            rep.fail(check_name,
                     f"{golden_dir} holds {len(npzs)} golden vectors "
                     f"({npzs}) -- each header/golden directory should hold "
                     f"exactly one model's artifacts, or this check cannot "
                     f"tell which one the header matches.")
            continue

        binary = os.path.join(workdir, f"full_model_{label}")
        res = subprocess.run(
            [cc, "-std=c99", "-O2", "-Wall", "-Wextra",
             "-I", os.path.join(ROOT, "sw", "include"),
             "-I", header_dir,
             os.path.join(ROOT, "train", "parity_src", "full_model_harness.c"),
             os.path.join(ROOT, "sw", "src", "quant.c"),
             os.path.join(ROOT, "sw", "src", "nn_baseline.c"),
             "-o", binary],
            capture_output=True, text=True)
        if res.returncode != 0:
            rep.fail(check_name,
                     "the generated header did not compile:\n" + res.stderr[:1500])
            continue

        run = subprocess.run([binary], capture_output=True, text=True)
        if run.returncode != 0:
            rep.fail(check_name, f"harness exited {run.returncode}")
            continue

        c_layers, c_class = [], None
        for line in run.stdout.splitlines():
            if line.startswith("LAYER"):
                c_layers.append([int(x) for x in line.split(":")[1].split()])
            elif line.startswith("CLASS"):
                c_class = int(line.split()[1])

        z = np.load(os.path.join(golden_dir, npzs[0]))
        keys = sorted([k for k in z.files if k.startswith("layer")],
                      key=lambda k: int(k.split("_")[0][5:]))

        if len(keys) != len(c_layers):
            rep.fail(check_name,
                     f"layer count differs: golden has {len(keys)}, C produced "
                     f"{len(c_layers)}. The topology table in the generated header "
                     f"does not match the config.")
            continue

        total = 0
        mismatched = False
        for i, k in enumerate(keys):
            exp = z[k].ravel().tolist()
            got = c_layers[i]
            total += len(exp)
            mm = first_mismatch(exp, got)
            if mm is not None:
                idx, e, g = mm
                rep.fail(check_name,
                         f"layer {i} ({k}) differs\n"
                         f"  first mismatch at element {idx}\n"
                         f"  python expected: {e}\n  C returned: {g}\n"
                         f"  ({len(exp)} elements in this layer)")
                mismatched = True
                break
        if mismatched:
            continue

        # Every layer matched, including the final one -- for a classifier
        # that already proves the class agrees, but nn_argmax's tie-breaking
        # rule is cheap to double-check explicitly. For an autoencoder this
        # is a tautological consistency check (both sides argmax the same
        # already-matched values), not a claim that the index means anything.
        golden_class = int(z["predicted_class"])
        if c_class != golden_class:
            rep.fail(check_name,
                     f"every layer matched but nn_argmax(final_layer) differs: "
                     f"python={golden_class} C={c_class}. Check nn_argmax's "
                     f"tie-breaking rule.")
            continue

        rep.ok(check_name, f"{len(keys)} layers, {total} values, "
                           f"final-layer argmax={c_class}")


# --------------------------------------------------------------------------
# Stage 2/3: Python vs Verilog, via Icarus
# --------------------------------------------------------------------------

RQ_TB = r"""
`timescale 1ns/1ps
// Auto-generated by train/verify_parity.py.
// Streams vectors through rtl/requantize.v and prints the results so Python
// can diff them. Kept dead simple: no clock, purely combinational.
module tb_parity_rq;
  reg signed [31:0] acc, mult, zp; reg [5:0] sh;
  wire signed [31:0] o8, o4;
  requantize #(.PRECISION(8)) u8 (.acc(acc),.mult(mult),.shift(sh),.zero_point(zp),.q_out(o8));
  requantize #(.PRECISION(4)) u4 (.acc(acc),.mult(mult),.shift(sh),.zero_point(zp),.q_out(o4));
  integer fd, r; reg [31:0] a_, m_, z_, s_, p_;
  initial begin
    fd = $fopen("VECFILE", "r");
    if (fd == 0) begin $display("ERROR cannot open vectors"); $finish(1); end
    while (!$feof(fd)) begin
      r = $fscanf(fd, "%h %h %d %h %d\n", a_, m_, s_, z_, p_);
      if (r == 5) begin
        acc = a_; mult = m_; sh = s_[5:0]; zp = z_;
        #1;
        if (p_ == 8) $display("%0d", o8); else $display("%0d", o4);
      end
    end
    $fclose(fd); $finish;
  end
endmodule
"""

DOT4_TB = r"""
`timescale 1ns/1ps
// Auto-generated by train/verify_parity.py.
// Drives rtl/dot4_pcpi.v through the real PCPI handshake and prints each
// DOT4 result, so Python can confirm the HARDWARE agrees with the reference.
module tb_parity_dot4;
  reg clk = 0, resetn = 0, valid = 0;
  reg [31:0] insn, rs1, rs2;
  wire wr, wait_, ready; wire [31:0] rd;
  always #5 clk = ~clk;
  dot4_pcpi dut (.clk(clk), .resetn(resetn), .pcpi_valid(valid),
    .pcpi_insn(insn), .pcpi_rs1(rs1), .pcpi_rs2(rs2), .pcpi_wr(wr),
    .pcpi_rd(rd), .pcpi_wait(wait_), .pcpi_ready(ready),
    .acc_value(), .acc_active());
  integer fd, r, guard; reg [31:0] a_, b_; reg signed [31:0] cap;
  initial begin
    fd = $fopen("VECFILE", "r");
    if (fd == 0) begin $display("ERROR cannot open vectors"); $finish(1); end
    @(posedge clk); resetn <= 1; @(posedge clk);
    while (!$feof(fd)) begin
      r = $fscanf(fd, "%h %h\n", a_, b_);
      if (r == 2) begin
        @(negedge clk);
        insn = {7'b0000000, 5'd6, 5'd5, 3'b000, 5'd7, 7'b0001011};
        rs1 = a_; rs2 = b_; valid = 1; guard = 0; cap = 0;
        while (!ready && guard < 64) begin @(posedge clk); #1; guard = guard + 1; end
        if (ready) cap = rd;
        $display("%0d", cap);
        @(negedge clk); valid = 0; insn = 0; @(negedge clk);
      end
    end
    $fclose(fd); $finish;
  end
endmodule
"""


def have_iverilog() -> bool:
    return shutil.which("iverilog") is not None and shutil.which("vvp") is not None


def run_iverilog(workdir: str, tb_src: str, tb_name: str,
                 rtl_files: list[str], vec_path: str) -> list[int]:
    tb_path = os.path.join(workdir, f"{tb_name}.v")
    with open(tb_path, "w") as fh:
        fh.write(tb_src.replace("VECFILE", vec_path))

    vvp_path = os.path.join(workdir, f"{tb_name}.vvp")
    cmd = ["iverilog", "-g2005", "-I", os.path.join(ROOT, "rtl"),
           "-o", vvp_path, tb_path] + rtl_files
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"iverilog failed:\n{res.stderr}")

    res = subprocess.run(["vvp", vvp_path], capture_output=True, text=True)
    out = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line and (line.lstrip("-").isdigit()):
            out.append(int(line))
    return out


def stage_python_vs_verilog(rep: Reporter, n: int, seed: int,
                            workdir: str) -> None:
    if not have_iverilog():
        rep.skip("Python vs Verilog",
                 "iverilog/vvp not found (brew install icarus-verilog, "
                 "or apt-get install iverilog)")
        return

    # --- requantize ---
    vecs = make_requant_vectors(n, seed)
    vec_path = os.path.join(workdir, "rq_vectors.txt")
    with open(vec_path, "w") as fh:
        for a, m, s, z, p in vecs:
            fh.write(f"{a & 0xFFFFFFFF:08x} {m:08x} {s:02d} "
                     f"{z & 0xFFFFFFFF:08x} {p}\n")

    expected = [int(qr.requantize(a, m, s, z, p)) for a, m, s, z, p in vecs]
    try:
        got = run_iverilog(workdir, RQ_TB, "tb_parity_rq",
                           [os.path.join(ROOT, "rtl", "requantize.v")], vec_path)
    except RuntimeError as exc:
        rep.fail("requantize: Python == Verilog", str(exc))
        return

    mm = first_mismatch(expected, got)
    if mm is None:
        rep.ok("requantize: Python == Verilog", f"{len(vecs)} vectors")
    else:
        i, e, g = mm
        a, m, s, z, p = vecs[i] if i >= 0 else (0,) * 5
        rep.fail("requantize: Python == Verilog",
                 f"first mismatch at index {i}\n"
                 f"  inputs: acc={a} m0={m} shift={s} zp={z} precision={p}\n"
                 f"  python expected: {e}\n  Verilog returned: {g}")

    # --- dot4 through the real PCPI handshake ---
    dvecs = make_dot4_vectors(min(n // 8, 200), seed)
    dvec_path = os.path.join(workdir, "dot4_vectors.txt")
    with open(dvec_path, "w") as fh:
        for a, b in dvecs:
            fh.write(f"{a:08x} {b:08x}\n")

    dexp = [qr.dot4(a, b, 8) for a, b in dvecs]
    try:
        dgot = run_iverilog(workdir, DOT4_TB, "tb_parity_dot4",
                            [os.path.join(ROOT, "rtl", "dot4_pcpi.v")], dvec_path)
    except RuntimeError as exc:
        rep.fail("dot4: Python == Verilog", str(exc))
        return

    mm = first_mismatch(dexp, dgot)
    if mm is None:
        rep.ok("dot4: Python == Verilog (via PCPI)", f"{len(dvecs)} vectors")
    else:
        i, e, g = mm
        a, b = dvecs[i] if i >= 0 else (0, 0)
        rep.fail("dot4: Python == Verilog (via PCPI)",
                 f"first mismatch at index {i}\n"
                 f"  inputs: a=0x{a:08x} b=0x{b:08x}\n"
                 f"  python expected: {e}\n  Verilog returned: {g}")


# --------------------------------------------------------------------------
# Stage 4: test the test
# --------------------------------------------------------------------------

def stage_self_test(rep: Reporter, workdir: str) -> None:
    """Inject a deliberate error and confirm the harness catches it.

    A parity harness that always passes is worse than no harness: it produces
    false confidence. Here we recompile the C with a flag that changes the
    rounding rule, and REQUIRE the comparison to fail. If it passes, the
    harness is not actually comparing anything.
    """
    sub = os.path.join(workdir, "selftest")
    os.makedirs(sub, exist_ok=True)

    # Rebuild the C with the rounding deliberately broken (floor instead of
    # away-from-zero on negatives). This is exactly the bug the negate trick
    # in quant.c exists to prevent, so it is a realistic fault to inject.
    broken_src = os.path.join(sub, "broken_quant.c")
    with open(os.path.join(ROOT, "sw", "src", "quant.c")) as fh:
        src = fh.read()
    src = src.replace("out = -(((-prod) + half) >> shift);",
                      "out = (prod + half) >> shift;  /* INJECTED FAULT */")
    with open(broken_src, "w") as fh:
        fh.write(src)

    harness_c = os.path.join(sub, "parity_harness.c")
    with open(harness_c, "w") as fh:
        fh.write(C_HARNESS)

    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        rep.skip("self-test (injected fault)", "no host C compiler")
        return

    binary = os.path.join(sub, "broken")
    res = subprocess.run(
        [cc, "-std=c99", "-O1", "-I", os.path.join(ROOT, "sw", "include"),
         harness_c, broken_src, "-o", binary],
        capture_output=True, text=True)
    if res.returncode != 0:
        rep.skip("self-test (injected fault)", "could not build the faulty variant")
        return

    # Negative accumulators with an exact-half result are where the injected
    # fault shows up. If the harness is working, these MUST differ.
    vecs = [(-1, 1 << 30, 31, 0, 8), (-3, 1 << 30, 31, 0, 8),
            (-5, 1 << 30, 31, 0, 8), (-7, 1 << 30, 31, 0, 8)]
    expected = [int(qr.requantize(a, m, s, z, p)) for a, m, s, z, p in vecs]
    got = run_c(binary, "r", [f"{a} {m} {s} {z} {p}" for a, m, s, z, p in vecs])

    if first_mismatch(expected, got) is not None:
        rep.ok("self-test: harness detects an injected rounding fault",
               f"python={expected} broken_c={got}")
    else:
        rep.fail("self-test: harness detects an injected rounding fault",
                 "The harness did NOT notice a deliberately broken rounding "
                 "rule.\nThat means it is not really comparing anything and "
                 "every other PASS above is meaningless.")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000,
                    help="number of random vectors per stage (default 2000)")
    ap.add_argument("--seed", type=int, default=20260828,
                    help="RNG seed; recorded so runs are reproducible")
    ap.add_argument("--stage", choices=["all", "c", "layers", "model", "verilog", "selftest"],
                    default="all", help="run only one stage")
    args = ap.parse_args()

    print("=" * 70)
    print(" BIT-EXACTNESS PARITY CHECK")
    print("=" * 70)
    print(f" seed={args.seed}  vectors={args.n}")
    print(" Comparing: train/quant_ref.py  <->  sw/src/quant.c  <->  rtl/*.v")
    print()

    rep = Reporter()
    with tempfile.TemporaryDirectory(prefix="parity_") as workdir:
        if args.stage in ("all", "c"):
            print(" Stage 1: Python vs C")
            stage_python_vs_c(rep, args.n, args.seed, workdir)
            print()
        if args.stage in ("all", "layers"):
            print(" Stage 2: layer kernels (Python vs C)")
            stage_layers(rep, args.seed, workdir)
            print()
        if args.stage in ("all", "model"):
            print(" Stage 3: the whole model (Python vs C)")
            stage_full_model(rep, workdir)
            print()
        if args.stage in ("all", "verilog"):
            print(" Stage 4: Python vs Verilog")
            stage_python_vs_verilog(rep, args.n, args.seed, workdir)
            print()
        if args.stage in ("all", "selftest"):
            print(" Stage 5: test the test")
            stage_self_test(rep, workdir)

    return rep.verdict()


if __name__ == "__main__":
    sys.exit(main())
