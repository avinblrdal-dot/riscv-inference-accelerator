/*===========================================================================
 * nn_dot4.c -- Inference using the DOT4 custom instruction (Phase 03)
 *===========================================================================
 *
 * Same maths as nn_baseline.c, same results BIT FOR BIT, but the innermost
 * loop processes four elements per instruction instead of one.
 *
 * WHAT CHANGED AND WHY IT IS FASTER
 * ---------------------------------
 * Baseline inner loop, per element:
 *     load, load, sign-extend, sign-extend, multiply, add, 2x increment,
 *     compare, branch          ~= 8-10 instructions for ONE multiply-add
 *
 * This version, per FOUR elements:
 *     load4, load4, dot4a, 2x increment, compare, branch
 *                              ~= 6 instructions for FOUR multiply-adds
 *
 * So roughly 6x fewer instructions per MAC. The real speedup will be lower
 * than 6x -- Amdahl's law applies, because requantization, pointer setup and
 * the outer loops are untouched. Measuring the gap between the theoretical
 * and achieved speedup is exactly the point of RQ2.
 *
 * BIT-EXACTNESS
 * -------------
 * dot4 sums its four products in a 32-bit accumulator, and integer addition
 * is associative, so grouping the reduction in fours cannot change the
 * result. That is what makes it legitimate to claim identical output rather
 * than merely similar output -- and train/verify_parity.py checks it rather
 * than taking it on trust.
 *
 * THE TAIL
 * --------
 * When the reduction length is not a multiple of 4, the leftover elements are
 * handled one at a time in plain C. Forgetting the tail is the classic
 * vectorisation bug: it passes every test whose dimensions happen to be a
 * multiple of 4 and silently drops data on everything else.
 *===========================================================================*/

#include "nn.h"
#include "accel.h"

/*---------------------------------------------------------------------------
 * nn_fc_dot4 -- fully-connected layer using DOT4A/ACCRD
 *-------------------------------------------------------------------------*/
void nn_fc_dot4(const int8_t *in, const nn_fc_t *layer, int8_t *out)
{
    for (int32_t o = 0; o < layer->out_dim; o++) {
        const int8_t *w = layer->weights + (int32_t)o * layer->in_dim;

        /* Clear the hardware accumulator before use. ACCRD is read-and-clear,
         * so this both zeroes it and discards anything a previous layer left
         * behind. Skipping this makes each output depend on the previous
         * one -- a bug that produces plausible-looking garbage. */
        (void)accrd();

        int32_t i = 0;
        const int32_t n4 = layer->in_dim & ~3;   /* largest multiple of 4 */

        /* ----- THE ACCELERATED MAC LOOP ----------------------------- */
        for (; i < n4; i += 4) {
            dot4a(load4(&in[i]), load4(&w[i]));
        }
        /* ------------------------------------------------------------ */

        int32_t acc = accrd() + (layer->bias ? layer->bias[o] : 0);

        /* ----- THE TAIL: 0-3 leftover elements ---------------------- */
        for (; i < layer->in_dim; i++) {
            acc += (int32_t)in[i] * (int32_t)w[i];
        }

        out[o] = (int8_t)nn_requantize(acc, layer->qp.multiplier,
                                       layer->qp.shift,
                                       layer->qp.zero_point,
                                       layer->qp.precision);
    }
}

/*---------------------------------------------------------------------------
 * nn_conv2d_dot4 -- convolution using DOT4
 *-------------------------------------------------------------------------*
 *
 * Convolution is harder to accelerate than a fully-connected layer, and the
 * reason is instructive: dot4 needs FOUR CONTIGUOUS bytes, but a convolution
 * window walks across a row and then jumps to the next row. Only the kw
 * elements within one row are contiguous.
 *
 * So we vectorise along the kernel WIDTH when kw >= 4, and fall back to
 * scalar otherwise. For the 3x3 kernels in our model, kw = 3 < 4, which means
 * the convolution layers get NO benefit from dot4 at all.
 *
 * That is not a defect to hide -- it is a finding, and an important one. It
 * says the custom-instruction approach helps fully-connected layers far more
 * than small-kernel convolutions, and it is precisely the motivation for the
 * MAC array in Phase 04, which reorganises the data (im2col) so that long
 * contiguous reductions exist to be accelerated. Report it in the paper.
 */
void nn_conv2d_dot4(const int8_t *in, int32_t in_h, int32_t in_w,
                    const nn_conv_t *layer, int8_t *out,
                    int32_t *out_h, int32_t *out_w, int32_t *scratch)
{
    (void)scratch;

    const int32_t kh = layer->kh, kw = layer->kw;
    const int32_t stride = layer->stride, pad = layer->pad;
    const int32_t in_ch = layer->in_ch, out_ch = layer->out_ch;

    const int32_t oh = (in_h + 2 * pad - kh) / stride + 1;
    const int32_t ow = (in_w + 2 * pad - kw) / stride + 1;

    const int32_t kw4 = kw & ~3;   /* how much of the row we can vectorise */

    for (int32_t oc = 0; oc < out_ch; oc++) {
        const int8_t *w_oc = layer->weights + (int32_t)oc * in_ch * kh * kw;
        const int32_t bias = layer->bias ? layer->bias[oc] : 0;

        for (int32_t oy = 0; oy < oh; oy++) {
            for (int32_t ox = 0; ox < ow; ox++) {
                int32_t acc = bias;

                for (int32_t ic = 0; ic < in_ch; ic++) {
                    const int8_t *w_ic = w_oc + (int32_t)ic * kh * kw;
                    const int8_t *x_ic = in + (int32_t)ic * in_h * in_w;

                    for (int32_t ky = 0; ky < kh; ky++) {
                        const int32_t iy = oy * stride + ky - pad;
                        if (iy < 0 || iy >= in_h) continue;

                        const int32_t ix0 = ox * stride - pad;

                        /* Fast path: the whole 4-wide group is inside the
                         * image, so no padding checks are needed and the
                         * bytes are contiguous. */
                        int32_t kx = 0;
                        if (ix0 >= 0 && ix0 + kw <= in_w) {
                            for (; kx < kw4; kx += 4) {
                                acc += dot4(load4(&x_ic[iy * in_w + ix0 + kx]),
                                            load4(&w_ic[ky * kw + kx]));
                            }
                        }
                        /* Scalar remainder, and the whole row when the window
                         * straddles a padded edge. */
                        for (; kx < kw; kx++) {
                            const int32_t ix = ix0 + kx;
                            if (ix < 0 || ix >= in_w) continue;
                            acc += (int32_t)x_ic[iy * in_w + ix] *
                                   (int32_t)w_ic[ky * kw + kx];
                        }
                    }
                }

                out[(oc * oh + oy) * ow + ox] =
                    (int8_t)nn_requantize(acc, layer->qp.multiplier,
                                          layer->qp.shift,
                                          layer->qp.zero_point,
                                          layer->qp.precision);
            }
        }
    }

    *out_h = oh;
    *out_w = ow;
}
