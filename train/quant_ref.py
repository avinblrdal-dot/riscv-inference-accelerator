"""Reference int8/int4 quantized arithmetic -- the single source of truth.

WHY THIS FILE EXISTS AND WHY IT USES ONLY NUMPY
-----------------------------------------------
The same arithmetic is implemented three times in this project:

    Python   train/quant_ref.py   (this file)   <-- the reference
    C        sw/src/quant.c
    Verilog  rtl/requantize.v

``train/verify_parity.py`` proves all three agree bit for bit on every golden
vector. That harness is the safety net for the entire project: without it, a
hardware bug and ordinary numerical drift look identical, and you cannot tell
whether a 2% accuracy drop means "int4 is too coarse" or "the sign extension
in the MAC array is broken".

This file deliberately depends on **numpy only**, never on PyTorch. Three
reasons:

1. The reference must be runnable anywhere -- CI, a laptop with no GPU, a
   machine where the torch install is broken. If proving bit-exactness
   required a 2 GB dependency, it would get skipped.
2. PyTorch's quantization backends change between releases and differ across
   CPU architectures. Pinning our semantics to *our own* explicit integer
   code means an upgrade cannot silently change the numbers.
3. It forces the arithmetic to be written out in full, which is what makes it
   translatable line by line into C and Verilog.

Everything here operates on int32/int64 numpy arrays. There is no floating
point anywhere in the inference path -- see ``requantize`` for why that is
not negotiable.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants -- keep in sync with rtl/accel_pkg.vh and sw/include/nn.h
# ---------------------------------------------------------------------------

#: The fixed-point multiplier M0 is normalized into [2^30, 2^31), so the
#: baseline right shift is 31. See ``quantize_multiplier``.
M0_SHIFT = 31


def qmin(precision: int) -> int:
    """Smallest representable value at the given precision (symmetric range)."""
    return -(1 << (precision - 1))


def qmax(precision: int) -> int:
    """Largest representable value at the given precision."""
    return (1 << (precision - 1)) - 1


def quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    """Convert a real scale factor into an integer (M0, shift) pair.

    The inference maths wants ``out = acc * real_multiplier`` where
    ``real_multiplier = (input_scale * weight_scale) / output_scale`` and lies
    in (0, 1). We cannot do that multiply in floating point (see
    ``requantize``), so we decompose it once, offline:

        real_multiplier = m * 2**e     with m in [0.5, 1) and e <= 0
        M0    = round(m * 2**31)       so M0 lands in [2^30, 2^31)
        shift = 31 - e                 (e is negative, so shift >= 31)

    and then at runtime ``acc * real_multiplier == (acc * M0) >> shift``.

    Returns:
        (M0, shift) as plain Python ints.

    Raises:
        ValueError: if the multiplier is outside the representable range,
            which means the model's scales are pathological and should be
            investigated rather than silently clamped.
    """
    if real_multiplier <= 0.0:
        raise ValueError(
            f"real_multiplier must be positive, got {real_multiplier}. "
            "A non-positive scale usually means a layer saw all-zero "
            "calibration data -- check the calibration set."
        )
    if real_multiplier >= 1.0:
        raise ValueError(
            f"real_multiplier must be < 1, got {real_multiplier}. "
            "Values >= 1 mean the output scale is smaller than the product of "
            "the input scales, which this integer pipeline does not support."
        )

    # np.frexp gives mantissa in [0.5, 1) and the exponent.
    mantissa, exponent = np.frexp(real_multiplier)
    m0 = int(round(float(mantissa) * (1 << M0_SHIFT)))

    # Edge case: a mantissa that rounds up to exactly 1.0 gives M0 == 2**31,
    # which does not fit in a signed 32-bit integer. Halve it and absorb the
    # factor of two into the exponent. Omitting this fix produces a negative
    # M0 in C and Verilog and flips the sign of an entire layer's output.
    if m0 == (1 << M0_SHIFT):
        m0 //= 2
        exponent += 1

    shift = M0_SHIFT - int(exponent)

    if not (1 << (M0_SHIFT - 1)) <= m0 < (1 << M0_SHIFT):
        raise ValueError(f"internal error: M0={m0} outside [2^30, 2^31)")
    if shift < 1 or shift > 62:
        raise ValueError(
            f"shift={shift} is outside the range the 64-bit datapath supports"
        )
    return m0, shift


def requantize(acc, m0: int, shift: int, zero_point: int = 0,
               precision: int = 8):
    """Scale an int32 accumulator down to `precision` bits, integer-only.

    THE ROUNDING RULE -- round half AWAY FROM ZERO, symmetrically:

        prod  = acc * m0                  (64-bit intermediate)
        half  = 1 << (shift - 1)
        acc >= 0:  out =  ( prod + half) >> shift
        acc <  0:  out = -((-prod + half) >> shift)

    The explicit negate-shift-negate for negatives is the important part. An
    arithmetic right shift rounds toward negative infinity, so the naive
    ``(prod + half) >> shift`` would round +2.5 to +3 but -2.5 to -2. That
    asymmetry injects a small positive DC bias into every layer, which
    accumulates through the network and costs real accuracy. Rounding away
    from zero on both sides is symmetric and unbiased.

    NOTE: this differs slightly from gemmlowp's exact "nudge" formulation, and
    that is a deliberate, documented choice -- see docs/DECISIONS.md,
    "Rounding rule". What matters scientifically is that our three
    implementations agree with each other and that the rule is written down,
    not that we reproduce Google's last bit.

    Args:
        acc: int32 accumulator(s); scalar or numpy array.
        m0: fixed-point multiplier from ``quantize_multiplier``.
        shift: right shift from ``quantize_multiplier``.
        zero_point: added after scaling (0 for the symmetric scheme we use).
        precision: 8 or 4.

    Returns:
        numpy int32 array (or scalar) clamped to the precision's range.
    """
    if shift < 1:
        raise ValueError(f"shift must be >= 1, got {shift}")

    # int64 throughout: acc is up to 2^31 and m0 up to 2^31, so the product
    # needs 62 bits. Doing this in int32 overflows on nearly every input and
    # is the single most destructive bug available in this file.
    a = np.asarray(acc, dtype=np.int64)
    prod = a * np.int64(m0)
    half = np.int64(1) << np.int64(shift - 1)

    # numpy's >> on negative int64 is an arithmetic shift (floor), which is
    # exactly what the negate trick below expects.
    pos = (prod + half) >> np.int64(shift)
    neg = -((-prod + half) >> np.int64(shift))
    shifted = np.where(prod >= 0, pos, neg)

    biased = shifted + np.int64(zero_point)
    clamped = np.clip(biased, qmin(precision), qmax(precision))
    return clamped.astype(np.int32)


def quantize_tensor(x: np.ndarray, scale: float, zero_point: int = 0,
                    precision: int = 8) -> np.ndarray:
    """Convert a float tensor to integers: ``q = clamp(round(x/scale) + zp)``.

    Uses round-half-away-from-zero to match ``requantize``. numpy's ``round``
    does banker's rounding (0.5 -> 0, 1.5 -> 2), which would disagree with the
    C and Verilog implementations on exact halves -- a rare but real source of
    single-LSB parity failures that is miserable to track down.
    """
    scaled = np.asarray(x, dtype=np.float64) / scale
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    q = rounded.astype(np.int64) + zero_point
    return np.clip(q, qmin(precision), qmax(precision)).astype(np.int32)


def dequantize_tensor(q: np.ndarray, scale: float,
                      zero_point: int = 0) -> np.ndarray:
    """Inverse of ``quantize_tensor``, for reporting and debugging only.

    Never call this inside the inference path -- it reintroduces floating
    point and would break bit-exactness with the hardware.
    """
    return (np.asarray(q, dtype=np.float64) - zero_point) * scale


def choose_scale(x: np.ndarray, precision: int = 8) -> float:
    """Pick a symmetric per-tensor scale for a float tensor.

    Symmetric means zero_point = 0, so the integer zero represents float zero
    exactly. That is worth having: padding, ReLU outputs and sparse
    activations are all exactly zero, and an asymmetric scheme would turn
    every one of them into a small nonzero number that then propagates.

    The scale maps the largest magnitude in the tensor onto the largest
    representable integer.
    """
    amax = float(np.max(np.abs(np.asarray(x, dtype=np.float64))))
    if amax == 0.0:
        # An all-zero tensor has no meaningful scale. Return 1.0 so the
        # arithmetic stays well-defined instead of producing div-by-zero or
        # NaN, and let the caller notice via the warning in quantize.py.
        return 1.0
    return amax / qmax(precision)


def dot4(a_word: int, b_word: int, precision: int = 8) -> int:
    """Reference model for the DOT4 custom instruction.

    Treats two 32-bit words as four packed signed lanes each and returns the
    4-way dot product. This is the golden model that rtl/dot4_pcpi.v and
    sw/include/accel.h are both checked against.
    """
    total = 0
    for lane in range(4):
        a = (a_word >> (lane * 8)) & 0xFF
        b = (b_word >> (lane * 8)) & 0xFF
        if precision == 4:
            a &= 0x0F
            b &= 0x0F
            a = a - 16 if a & 0x8 else a
            b = b - 16 if b & 0x8 else b
        else:
            a = a - 256 if a & 0x80 else a
            b = b - 256 if b & 0x80 else b
        total += a * b
    return total


def conv2d_int(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None,
               stride: int = 1, pad: int = 0) -> np.ndarray:
    """Integer 2-D convolution producing int32 accumulators.

    Shapes:
        x: (C_in, H, W)          int8-valued
        w: (C_out, C_in, kH, kW) int8-valued
        b: (C_out,)              int32 bias, already in accumulator units

    No requantization happens here -- this returns the raw int32 sums so the
    caller can apply ``requantize`` with the right per-layer multiplier. That
    split mirrors the hardware, where the MAC array accumulates and a separate
    unit rescales.

    Computed in int64 internally and returned as int32. The int64 is not
    paranoia: 3x3x16 channels of -128*-128 is about 2.4 million, and deeper
    layers with wide reductions get much closer to the int32 limit than people
    expect.
    """
    c_in, h_in, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    if c_in != c_in_w:
        raise ValueError(
            f"channel mismatch: input has {c_in}, weights expect {c_in_w}"
        )

    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")

    h_out = (x.shape[1] - kh) // stride + 1
    w_out = (x.shape[2] - kw) // stride + 1
    out = np.zeros((c_out, h_out, w_out), dtype=np.int64)

    xi = x.astype(np.int64)
    wi = w.astype(np.int64)

    for oc in range(c_out):
        for oy in range(h_out):
            for ox in range(w_out):
                patch = xi[:, oy*stride:oy*stride+kh, ox*stride:ox*stride+kw]
                out[oc, oy, ox] = int(np.sum(patch * wi[oc]))
        if b is not None:
            out[oc] += np.int64(b[oc])

    return out.astype(np.int32)


def linear_int(x: np.ndarray, w: np.ndarray,
               b: np.ndarray | None = None) -> np.ndarray:
    """Integer fully-connected layer: ``out[o] = sum_i x[i]*w[o][i] + b[o]``.

    Shapes: x is (N,), w is (M, N), b is (M,). Returns int32 accumulators.
    """
    xi = np.asarray(x, dtype=np.int64)
    wi = np.asarray(w, dtype=np.int64)
    out = wi @ xi
    if b is not None:
        out = out + np.asarray(b, dtype=np.int64)
    return out.astype(np.int32)


def relu_int(x: np.ndarray, zero_point: int = 0) -> np.ndarray:
    """ReLU in the integer domain: clamp at the value representing zero."""
    return np.maximum(np.asarray(x), zero_point)


def maxpool2d_int(x: np.ndarray, k: int = 2, stride: int | None = None) -> np.ndarray:
    """Integer max pooling over (C, H, W).

    Max pooling is exact in the quantized domain -- it only ever selects an
    existing value, never combines them -- so no requantization is needed and
    no error is introduced. Average pooling would NOT have that property.
    """
    if stride is None:
        stride = k
    c, h, w = x.shape
    h_out = (h - k) // stride + 1
    w_out = (w - k) // stride + 1
    out = np.zeros((c, h_out, w_out), dtype=x.dtype)
    for ch in range(c):
        for oy in range(h_out):
            for ox in range(w_out):
                out[ch, oy, ox] = np.max(
                    x[ch, oy*stride:oy*stride+k, ox*stride:ox*stride+k]
                )
    return out
