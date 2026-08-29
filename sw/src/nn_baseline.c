/*===========================================================================
 * nn_baseline.c -- Plain C inference. NO acceleration. The control condition.
 *===========================================================================
 *
 * This file is the scientific baseline. Every speedup and every energy saving
 * this project reports is measured against it, so it has two jobs that pull
 * in opposite directions:
 *
 *   1. Be HONEST. It must be a competent, ordinary implementation -- the kind
 *      an embedded engineer would actually write. Making the baseline
 *      artificially slow would inflate every result and would be a form of
 *      scientific misconduct, even if unintentional. Concretely: the loops
 *      below are ordered for cache/register friendliness, the innermost loop
 *      hoists what it can, and it is compiled at -O2 like everything else.
 *
 *   2. Be UNACCELERATED. No dot4, no MAC array, no hand-written assembly.
 *      Just C that any rv32i core can run.
 *
 * WHY THE MAC LOOP IS SO EXPENSIVE HERE (this is RQ1)
 * ---------------------------------------------------
 * The inner loop below does ONE useful multiply-add. Around it, rv32i needs
 * roughly: two loads, two sign-extensions, a multiply (which rv32i lacks, so
 * it is a called routine or a shift-add sequence), an add, a pointer
 * increment, a compare and a branch. That is the ~6-10 instructions of
 * overhead per useful MAC quoted in the README, and it is structural: the
 * instruction set simply has no way to express "do four of these at once".
 *
 * Instrumenting this file with perf.h is how we measure the fraction of total
 * cycles spent in MAC work, which sets the Amdahl ceiling for the whole
 * project.
 *===========================================================================*/

#include "nn.h"

/*---------------------------------------------------------------------------
 * nn_conv2d -- quantized 2-D convolution
 *-------------------------------------------------------------------------*
 *
 * in:      (in_ch, in_h, in_w) int8, row-major
 * out:     (out_ch, out_h, out_w) int8, row-major -- caller supplies storage
 * scratch: at least out_h*out_w int32 -- not used here, but kept in the
 *          signature so the accelerated variants (which need a place to land
 *          int32 accumulators) can share the same prototype and be swapped
 *          in without touching main.c.
 *
 * The accumulator is int32 and requantization happens once per output pixel,
 * at the end -- never inside the reduction. Requantizing early would throw
 * away precision on every step and is a classic accuracy bug.
 */
void nn_conv2d(const int8_t *in, int32_t in_h, int32_t in_w,
               const nn_conv_t *layer, int8_t *out,
               int32_t *out_h, int32_t *out_w, int32_t *scratch)
{
    (void)scratch;   /* unused in the baseline; see the note above */

    const int32_t kh = layer->kh, kw = layer->kw;
    const int32_t stride = layer->stride, pad = layer->pad;
    const int32_t in_ch = layer->in_ch, out_ch = layer->out_ch;

    const int32_t oh = (in_h + 2 * pad - kh) / stride + 1;
    const int32_t ow = (in_w + 2 * pad - kw) / stride + 1;

    for (int32_t oc = 0; oc < out_ch; oc++) {
        /* Hoisted out of the spatial loops: the weight base for this output
         * channel does not change as we slide the window. */
        const int8_t *w_oc = layer->weights + (int32_t)oc * in_ch * kh * kw;
        const int32_t bias = layer->bias ? layer->bias[oc] : 0;

        for (int32_t oy = 0; oy < oh; oy++) {
            for (int32_t ox = 0; ox < ow; ox++) {

                int32_t acc = bias;

                /* ----- THE MAC LOOP -- the thing this project is about ---
                 * Everything below is one multiply-accumulate surrounded by
                 * address arithmetic and loop control. Count the instructions
                 * the compiler emits for this and you have RQ1's answer. */
                for (int32_t ic = 0; ic < in_ch; ic++) {
                    const int8_t *w_ic = w_oc + (int32_t)ic * kh * kw;
                    const int8_t *x_ic = in + (int32_t)ic * in_h * in_w;

                    for (int32_t ky = 0; ky < kh; ky++) {
                        const int32_t iy = oy * stride + ky - pad;
                        /* Skip whole rows that fall outside the padded edge,
                         * rather than testing every column. Zero padding
                         * contributes nothing to the sum by definition, so
                         * skipping is exact, not an approximation. */
                        if (iy < 0 || iy >= in_h) continue;

                        for (int32_t kx = 0; kx < kw; kx++) {
                            const int32_t ix = ox * stride + kx - pad;
                            if (ix < 0 || ix >= in_w) continue;

                            acc += (int32_t)x_ic[iy * in_w + ix] *
                                   (int32_t)w_ic[ky * kw + kx];
                        }
                    }
                }
                /* ----- end MAC loop ----------------------------------- */

                /* One requantization per output element, after the full
                 * reduction. */
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

/*---------------------------------------------------------------------------
 * nn_fc -- fully-connected layer
 *-------------------------------------------------------------------------*/
void nn_fc(const int8_t *in, const nn_fc_t *layer, int8_t *out)
{
    for (int32_t o = 0; o < layer->out_dim; o++) {
        const int8_t *w = layer->weights + (int32_t)o * layer->in_dim;
        int32_t acc = layer->bias ? layer->bias[o] : 0;

        /* ----- THE MAC LOOP (fully-connected form) ------------------- */
        for (int32_t i = 0; i < layer->in_dim; i++) {
            acc += (int32_t)in[i] * (int32_t)w[i];
        }
        /* ------------------------------------------------------------- */

        out[o] = (int8_t)nn_requantize(acc, layer->qp.multiplier,
                                       layer->qp.shift,
                                       layer->qp.zero_point,
                                       layer->qp.precision);
    }
}
