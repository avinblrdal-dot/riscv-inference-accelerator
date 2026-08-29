/*===========================================================================
 * nn_array.c -- Inference driving the memory-mapped MAC array (Phase 04)
 *===========================================================================
 *
 * THE OPERAND CONTRACT -- read this before changing anything here.
 *
 * The array is output-stationary. Cell (i,j) owns one output element and
 * accumulates over k:
 *
 *     C[i][j] = sum over k of  A[i][k] * B[k][j]
 *
 * accel_top.v fans the 32-bit buffer words out to the cells like this:
 *
 *     row i    reads ACTIVATION lane (i % 4)
 *     column j reads WEIGHT     lane (j % 4)
 *
 * So a buffer word does NOT hold four consecutive values of one vector. It
 * holds ONE value for each of four different rows (or columns), at the same k.
 * Concretely, for a single input vector (M = 1) and four output channels:
 *
 *     activation word at address k :  { -, -, -, a[k] }        (lane 0 only)
 *     weight word     at address k :  { w[o+3][k], w[o+2][k],
 *                                       w[o+1][k], w[o+0][k] }
 *     dimensions                   :  M = 1, N = 4, K = in_dim
 *
 * and cell (0,j) then computes sum_k a[k] * w[o+j][k] -- four outputs at once.
 *
 * A PREVIOUS VERSION OF THIS FILE GOT THIS WRONG, and it is worth recording
 * exactly how, because the failure was silent. It packed four CONSECUTIVE
 * elements per word -- pack4(w[k], w[k+1], w[k+2], w[k+3]) -- and set
 * K = in_dim/4, expecting the hardware to perform a 4-way dot product per
 * word the way the DOT4 instruction does. It does not. With M = N = 1 only
 * cell (0,0) is read, and that cell sees lane 0 of each word, so the
 * accumulator became
 *
 *     sum over words w of  a[4w] * b[4w]
 *
 * -- every fourth element, three quarters of the data discarded. The model
 * still ran, still terminated, still produced a confident classification, and
 * was 5.1x faster than the baseline. It was simply the WRONG ANSWER
 * (predicted class 1 against a golden value of 2).
 *
 * That is the exact failure mode docs/REVIEW.md flags as risk R5: a
 * divergence in the accelerated path is indistinguishable from an accuracy
 * result unless something checks it. Treat any accuracy difference between
 * variants as a bug until proven otherwise -- the variants must agree
 * BIT-EXACTLY, not approximately.
 *
 * TWO FURTHER CONSTRAINTS THIS FILE MUST RESPECT
 *
 *   1. K is bounded by the buffer depth. WBUF_DEPTH is 256 words by default,
 *      but the fully-connected layer has in_dim = 1024. Pushing 1024 words
 *      wraps the write pointer and silently overwrites earlier weights, so
 *      the reduction is tiled over k and the partial sums are accumulated in
 *      software.
 *   2. The geometry is discovered at run time, not assumed. The sweep builds
 *      many array shapes from one firmware image; hardcoding 4x4 would
 *      silently miscompute on every other configuration.
 *===========================================================================*/

#include "nn.h"
#include "accel.h"

/* Bounded, because an unbounded poll turns a hardware fault into a silent
 * hang with no output at all. */
#define ACCEL_POLL_LIMIT 4000000u

/* Largest reduction length pushed in one pass. Must not exceed the smaller of
 * WBUF_DEPTH and ABUF_DEPTH, or the buffer write pointer wraps and quietly
 * corrupts the tile. 64 is the smallest nonzero depth in the sweep, so this
 * is safe for every configuration including the DEPTH=0 control case (which
 * bypasses storage entirely and is unaffected by depth). */
#define ACCEL_KTILE 64

/* Receptive field scratch for im2col. Static, not stack: link.ld gives only
 * 4 KB of stack. */
#define MAX_PATCH 512
static int8_t patch_buf[MAX_PATCH];

/* Accumulated across a run so main.c can report utilisation. */
uint32_t g_accel_total_cycles;
uint32_t g_accel_total_stalls;
uint32_t g_accel_total_macs;
int32_t  g_accel_timeouts;

/*---------------------------------------------------------------------------
 * accel_dot_group -- the one primitive everything else is built on
 *-------------------------------------------------------------------------*
 *
 * Computes, for up to `lanes` output channels at once:
 *
 *     acc[j] += sum over k in [0, len) of  in[k] * w_rows[j][k]
 *
 * `w_rows[j]` is a pointer to output channel (o0 + j)'s weight vector.
 * Results are ADDED into acc[], so the caller can tile over k.
 */
static void accel_dot_group(const int8_t *in,
                            const int8_t *const *w_rows,
                            int32_t lanes,
                            int32_t len,
                            int32_t *acc)
{
    int32_t k0;

    for (k0 = 0; k0 < len; k0 += ACCEL_KTILE) {
        int32_t klen = len - k0;
        int32_t k;
        if (klen > ACCEL_KTILE) klen = ACCEL_KTILE;

        /* Rewind the buffer write pointers. Without this each tile appends
         * after the previous one and runs off the end of the buffer. */
        ACCEL_REG_CTRL = ACCEL_CTRL_SOFT_RESET;

        for (k = 0; k < klen; k++) {
            /* Row 0 only (M = 1), so the activation goes in lane 0. The other
             * lanes feed rows 1..3, which this mapping does not use. */
            accel_push_activation(pack4(in[k0 + k], 0, 0, 0));

            /* Lane j feeds column j, i.e. output channel o0 + j. Absent
             * channels are padded with zero, which contributes nothing to a
             * dot product -- exact, not approximate, because the quantization
             * is symmetric so integer zero really is float zero. */
            accel_push_weight(pack4(
                (lanes > 0) ? w_rows[0][k0 + k] : 0,
                (lanes > 1) ? w_rows[1][k0 + k] : 0,
                (lanes > 2) ? w_rows[2][k0 + k] : 0,
                (lanes > 3) ? w_rows[3][k0 + k] : 0));
        }

        accel_set_dims(1u, (uint32_t)lanes, (uint32_t)klen);
        accel_start();

        if (accel_wait_done(ACCEL_POLL_LIMIT) != 0) {
            /* Record and continue with a defined value. A visible wrong
             * answer beats a silent hang; main.c reports the count and
             * cycle_capture.py treats a nonzero count as invalidating the run. */
            g_accel_timeouts++;
            continue;
        }

        /* The controller drains one whole tile (ARRAY_H * ARRAY_W elements)
         * into the FIFO in row-major order. Our results are row 0, columns
         * 0..lanes-1, so they are the first `lanes` entries. Everything after
         * that belongs to rows we did not use and must be popped and
         * discarded, or it would be misread as the next tile's results. */
        {
            const int32_t tile_elems =
                (int32_t)(accel_array_h() * accel_array_w());
            int32_t e;
            for (e = 0; e < tile_elems; e++) {
                const int32_t v = accel_pop_result();
                if (e < lanes) acc[e] += v;
            }
        }
    }
}

/*---------------------------------------------------------------------------
 * nn_fc_array -- fully-connected layer on the MAC array
 *-------------------------------------------------------------------------*
 *
 * M = 1 because a sensor node classifies one window at a time; there is no
 * batch to amortise over. That means only ONE ROW of the array does useful
 * work, so a tall array is mostly idle here. That is a genuine limitation of
 * batch-size-1 inference and belongs in the write-up rather than being
 * hidden: the array's advantage over dot4 on fully-connected layers is
 * smaller than a raw MACs-per-cycle count suggests.
 */
void nn_fc_array(const int8_t *in, const nn_fc_t *layer, int8_t *out)
{
    const int32_t lanes_max = (int32_t)accel_lanes();
    int32_t o;

    accel_clear_perf();

    for (o = 0; o < layer->out_dim; o += lanes_max) {
        const int8_t *w_rows[4];
        int32_t acc[4];
        int32_t lanes = layer->out_dim - o;
        int32_t j;

        if (lanes > lanes_max) lanes = lanes_max;

        for (j = 0; j < 4; j++) {
            acc[j] = 0;
            w_rows[j] = (j < lanes)
                ? layer->weights + (int32_t)(o + j) * layer->in_dim
                : layer->weights;   /* unused; never read when j >= lanes */
        }

        accel_dot_group(in, w_rows, lanes, layer->in_dim, acc);

        for (j = 0; j < lanes; j++) {
            const int32_t biased =
                acc[j] + (layer->bias ? layer->bias[o + j] : 0);
            out[o + j] = (int8_t)nn_requantize(biased,
                                               layer->qp.multiplier,
                                               layer->qp.shift,
                                               layer->qp.zero_point,
                                               layer->qp.precision);
        }
    }

    g_accel_total_cycles += accel_cycles();
    g_accel_total_stalls += accel_stalls();
    g_accel_total_macs   += accel_macs();
}

/*---------------------------------------------------------------------------
 * nn_conv2d_array -- convolution via im2col, one output position at a time
 *-------------------------------------------------------------------------*
 *
 * A convolution's sliding window is not contiguous in memory, which is what
 * stopped the DOT4 instruction from helping (see nn_dot4.c). Unfolding one
 * receptive field into a contiguous vector turns the problem into exactly the
 * fully-connected case above, with in_dim = in_ch * kh * kw -- a long
 * reduction, which is what the array wants.
 *
 * We unfold ONE position at a time rather than the whole layer. Full im2col
 * would expand the image by roughly kh*kw, and with 64 KB of RAM that does
 * not fit.
 */
void nn_conv2d_array(const int8_t *in, int32_t in_h, int32_t in_w,
                     const nn_conv_t *layer, int8_t *out,
                     int32_t *out_h, int32_t *out_w, int32_t *scratch)
{
    (void)scratch;

    const int32_t kh = layer->kh, kw = layer->kw;
    const int32_t stride = layer->stride, pad = layer->pad;
    const int32_t in_ch = layer->in_ch, out_ch = layer->out_ch;
    const int32_t patch_len = in_ch * kh * kw;
    const int32_t lanes_max = (int32_t)accel_lanes();

    const int32_t oh = (in_h + 2 * pad - kh) / stride + 1;
    const int32_t ow = (in_w + 2 * pad - kw) / stride + 1;

    int32_t oy, ox, oc, ic, ky, kx, j;

    *out_h = oh;
    *out_w = ow;

    if (patch_len > MAX_PATCH) {
        /* Fail predictably rather than smashing memory. Zeros are obviously
         * wrong downstream, which is the intent. */
        int32_t i;
        g_accel_timeouts++;
        for (i = 0; i < out_ch * oh * ow; i++) out[i] = 0;
        return;
    }

    accel_clear_perf();

    for (oy = 0; oy < oh; oy++) {
        for (ox = 0; ox < ow; ox++) {

            /* ---- im2col for ONE output position ---------------------- */
            int32_t p = 0;
            for (ic = 0; ic < in_ch; ic++) {
                const int8_t *x_ic = in + (int32_t)ic * in_h * in_w;
                for (ky = 0; ky < kh; ky++) {
                    const int32_t iy = oy * stride + ky - pad;
                    for (kx = 0; kx < kw; kx++) {
                        const int32_t ix = ox * stride + kx - pad;
                        /* Outside the image contributes zero. Exact under
                         * symmetric quantization. */
                        patch_buf[p++] = (iy < 0 || iy >= in_h ||
                                          ix < 0 || ix >= in_w)
                                         ? 0 : x_ic[iy * in_w + ix];
                    }
                }
            }

            /* ---- the unfolded window is now a fully-connected layer --- */
            for (oc = 0; oc < out_ch; oc += lanes_max) {
                const int8_t *w_rows[4];
                int32_t acc[4];
                int32_t lanes = out_ch - oc;

                if (lanes > lanes_max) lanes = lanes_max;

                for (j = 0; j < 4; j++) {
                    acc[j] = 0;
                    w_rows[j] = (j < lanes)
                        ? layer->weights + (int32_t)(oc + j) * patch_len
                        : layer->weights;
                }

                accel_dot_group(patch_buf, w_rows, lanes, patch_len, acc);

                for (j = 0; j < lanes; j++) {
                    const int32_t biased =
                        acc[j] + (layer->bias ? layer->bias[oc + j] : 0);
                    out[((oc + j) * oh + oy) * ow + ox] =
                        (int8_t)nn_requantize(biased,
                                              layer->qp.multiplier,
                                              layer->qp.shift,
                                              layer->qp.zero_point,
                                              layer->qp.precision);
                }
            }
        }
    }

    g_accel_total_cycles += accel_cycles();
    g_accel_total_stalls += accel_stalls();
    g_accel_total_macs   += accel_macs();
}
