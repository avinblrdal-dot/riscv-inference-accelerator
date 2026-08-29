/*===========================================================================
 * quant.c -- Integer quantized arithmetic (the C half of the parity triangle)
 *===========================================================================
 *
 * Every function here has an exact counterpart in train/quant_ref.py and,
 * for requantize, in rtl/requantize.v. train/verify_parity.py checks all
 * three agree on every golden vector. Treat a parity failure as a STOP-THE-
 * LINE event: it means one of the three has drifted, and until you know which,
 * no measurement from this project can be trusted.
 *===========================================================================*/

#include "nn.h"

/*---------------------------------------------------------------------------
 * nn_saturate -- clamp into the representable range
 *-------------------------------------------------------------------------*/
/* Saturation, not wraparound. If a value overflows we want the nearest
 * representable number. Wrapping would turn a large positive activation into
 * a large negative one and produce a confidently wrong classification -- and,
 * being data-dependent, it would only show up on some inputs. */
int32_t nn_saturate(int32_t v, int32_t precision)
{
    const int32_t lo = NN_QMIN(precision);
    const int32_t hi = NN_QMAX(precision);
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/*---------------------------------------------------------------------------
 * nn_requantize -- int32 accumulator -> `precision` bits, integer only
 *-------------------------------------------------------------------------*
 *
 * The rule, identical to quant_ref.requantize() and rtl/requantize.v:
 *
 *     prod  = (int64)acc * (int64)multiplier
 *     half  = 1 << (shift - 1)
 *     prod >= 0:  out =  ( prod + half) >> shift
 *     prod <  0:  out = -((-prod + half) >> shift)
 *     out  += zero_point, then saturate
 *
 * TWO THINGS THAT LOOK LIKE STYLE BUT ARE NOT:
 *
 * 1. The int64_t intermediate. acc can be 2^31 and multiplier is close to
 *    2^31, so the product needs 62 bits. Computing it in int32_t overflows
 *    on almost every input. This is the single most destructive possible bug
 *    in this file and it is completely silent.
 *
 * 2. The explicit negate-shift-negate on the negative branch. In C, >> on a
 *    negative signed value is implementation-defined before C++20 and is an
 *    arithmetic (floor) shift on every compiler we care about. Floor rounding
 *    would make -2.5 round to -2 while +2.5 rounds to +3 -- asymmetric, which
 *    injects a positive DC bias into every layer. Negating first makes the
 *    rounding symmetric about zero.
 */
int32_t nn_requantize(int32_t acc, int32_t multiplier, int32_t shift,
                      int32_t zero_point, int32_t precision)
{
    int64_t prod = (int64_t)acc * (int64_t)multiplier;
    int64_t half = (int64_t)1 << (shift - 1);
    int64_t out;

    if (prod >= 0) {
        out = (prod + half) >> shift;
    } else {
        out = -(((-prod) + half) >> shift);
    }

    out += (int64_t)zero_point;

    /* Clamp in 64-bit BEFORE narrowing to int32. Narrowing first would wrap
     * a huge accumulator around and then clamp the wrong value. */
    {
        const int64_t lo = (int64_t)NN_QMIN(precision);
        const int64_t hi = (int64_t)NN_QMAX(precision);
        if (out < lo) out = lo;
        if (out > hi) out = hi;
    }
    return (int32_t)out;
}

/*---------------------------------------------------------------------------
 * nn_relu
 *-------------------------------------------------------------------------*/
/* In the quantized domain, "zero" is whatever integer represents float zero.
 * With our symmetric scheme that is literally 0, but taking zero_point as a
 * parameter keeps the code correct if anyone later switches to an asymmetric
 * scheme -- a change that otherwise breaks ReLU silently. */
void nn_relu(int8_t *x, int32_t n, int32_t zero_point)
{
    for (int32_t i = 0; i < n; i++) {
        if (x[i] < zero_point) {
            x[i] = (int8_t)zero_point;
        }
    }
}

/*---------------------------------------------------------------------------
 * nn_maxpool2d
 *-------------------------------------------------------------------------*/
/* Max pooling is exact in the quantized domain: it only ever SELECTS an
 * existing value, so no requantization is needed and no error is introduced.
 * Average pooling would not have that property -- it would need a divide and
 * a rescale, and would be another place for the three implementations to
 * disagree. That is part of why the model uses max pooling. */
void nn_maxpool2d(const int8_t *in, int32_t ch, int32_t h, int32_t w,
                  int32_t k, int8_t *out, int32_t *out_h, int32_t *out_w)
{
    const int32_t oh = (h - k) / k + 1;
    const int32_t ow = (w - k) / k + 1;

    for (int32_t c = 0; c < ch; c++) {
        for (int32_t oy = 0; oy < oh; oy++) {
            for (int32_t ox = 0; ox < ow; ox++) {
                int8_t best = in[(c * h + oy * k) * w + ox * k];
                for (int32_t ky = 0; ky < k; ky++) {
                    for (int32_t kx = 0; kx < k; kx++) {
                        const int8_t v =
                            in[(c * h + oy * k + ky) * w + ox * k + kx];
                        if (v > best) best = v;
                    }
                }
                out[(c * oh + oy) * ow + ox] = best;
            }
        }
    }
    *out_h = oh;
    *out_w = ow;
}

/*---------------------------------------------------------------------------
 * nn_argmax
 *-------------------------------------------------------------------------*/
/* Ties go to the LOWEST index. That is an arbitrary choice, but it must be
 * the SAME arbitrary choice in Python, C and any hardware post-processing, or
 * the three will occasionally disagree on a genuinely tied output and the
 * parity harness will fail in a way that looks like a numerical bug. */
int32_t nn_argmax(const int8_t *x, int32_t n)
{
    int32_t best_i = 0;
    int8_t  best_v = x[0];
    for (int32_t i = 1; i < n; i++) {
        if (x[i] > best_v) {
            best_v = x[i];
            best_i = i;
        }
    }
    return best_i;
}
