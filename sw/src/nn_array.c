/*===========================================================================
 * nn_array.c -- Inference driving the memory-mapped MAC array (Phase 04)
 *===========================================================================
 *
 * The MAC array computes a tile of a matrix multiply: C[M][N] = A[M][K]*B[K][N].
 * Neural network layers are not written that way, so this file's real job is
 * RESHAPING -- turning convolutions and fully-connected layers into matrix
 * multiplies the array can eat.
 *
 * WHY THIS IS DIFFERENT FROM nn_dot4.c
 * ------------------------------------
 * dot4 is synchronous: the core issues the instruction and stalls until the
 * answer comes back. The array is asynchronous: the core writes operands,
 * says "go", and is then free. That freedom is the whole point -- but it also
 * means the core must now think about DATA MOVEMENT, which is what RQ3 is
 * about. Most of the code below is not arithmetic; it is getting bytes to the
 * right place, and that is exactly the cost the experiment is designed to
 * expose.
 *
 * im2col, briefly
 * ---------------
 * A convolution slides a window over an image. If you unfold every window
 * position into a COLUMN of a big matrix, the convolution becomes one matrix
 * multiply. The cost is memory: overlapping windows duplicate data, so the
 * unfolded matrix is larger than the image by roughly kh*kw. On a machine
 * with 64 KB of RAM that is a real constraint, which is why the code below
 * unfolds ONE TILE AT A TIME rather than the whole layer.
 *
 * THE PERFORMANCE COUNTERS
 * ------------------------
 * After each layer, accel_stalls() and accel_macs() are read out. Those two
 * numbers are the evidence for RQ3: if stalls dominate, the array is starved
 * and adding width will not help -- a falsifiable prediction.
 *===========================================================================*/

#include "nn.h"
#include "accel.h"

/* How long to wait for a job before declaring the hardware wedged. Sized
 * generously (a full tile at the widest reduction we use is well under this)
 * but finite, because an infinite wait turns a hardware bug into a silent
 * hang with no output at all. */
#define ACCEL_POLL_LIMIT 2000000u

/* Accumulated across a run so main.c can report utilisation. */
uint32_t g_accel_total_cycles;
uint32_t g_accel_total_stalls;
uint32_t g_accel_total_macs;
int32_t  g_accel_timeouts;

/*---------------------------------------------------------------------------
 * nn_fc_array -- fully-connected layer on the MAC array
 *-------------------------------------------------------------------------*
 *
 * A fully-connected layer IS a matrix multiply already:
 *     out[1][out_dim] = in[1][in_dim] * W_transposed[in_dim][out_dim]
 * so M = 1, N = out_dim, K = in_dim. No im2col needed.
 *
 * M = 1 means only one ROW of the array does useful work, which wastes most
 * of a tall array. That is a genuine limitation of batch-size-1 inference and
 * worth stating plainly: it is why the array's advantage over dot4 is smaller
 * on fully-connected layers than a naive MACs-per-cycle count suggests.
 * Batching would fix it, but a sensor node classifies one window at a time,
 * so batching is not available to us. Report this rather than hiding it.
 */
void nn_fc_array(const int8_t *in, const nn_fc_t *layer, int8_t *out)
{
    accel_clear_perf();

    /* Push the activation vector once -- it is reused for every output. */
    int32_t i = 0;
    for (; i + 4 <= layer->in_dim; i += 4) {
        accel_push_activation(pack4(in[i], in[i+1], in[i+2], in[i+3]));
    }
    if (i < layer->in_dim) {
        /* Pad the final partial word with zeros. Zeros contribute nothing to
         * a dot product, so this is exact rather than an approximation --
         * but only because our quantization is SYMMETRIC (zero_point == 0).
         * With an asymmetric scheme the pad value would have to be the zero
         * point instead, and using 0 would inject a real error. */
        int8_t t[4] = {0, 0, 0, 0};
        for (int32_t j = 0; i + j < layer->in_dim; j++) t[j] = in[i + j];
        accel_push_activation(pack4(t[0], t[1], t[2], t[3]));
    }

    for (int32_t o = 0; o < layer->out_dim; o++) {
        const int8_t *w = layer->weights + (int32_t)o * layer->in_dim;

        int32_t k = 0;
        for (; k + 4 <= layer->in_dim; k += 4) {
            accel_push_weight(pack4(w[k], w[k+1], w[k+2], w[k+3]));
        }
        if (k < layer->in_dim) {
            int8_t t[4] = {0, 0, 0, 0};
            for (int32_t j = 0; k + j < layer->in_dim; j++) t[j] = w[k + j];
            accel_push_weight(pack4(t[0], t[1], t[2], t[3]));
        }

        accel_set_dims(1, 1, (uint32_t)((layer->in_dim + 3) / 4));
        accel_start();

        if (accel_wait_done(ACCEL_POLL_LIMIT) != 0) {
            /* Do not spin forever. Record it, produce a defined output, and
             * let main.c report the count -- a visible wrong answer beats a
             * silent hang every time. */
            g_accel_timeouts++;
            out[o] = 0;
            continue;
        }

        int32_t acc = accel_pop_result() + (layer->bias ? layer->bias[o] : 0);

        out[o] = (int8_t)nn_requantize(acc, layer->qp.multiplier,
                                       layer->qp.shift,
                                       layer->qp.zero_point,
                                       layer->qp.precision);
    }

    g_accel_total_cycles += accel_cycles();
    g_accel_total_stalls += accel_stalls();
    g_accel_total_macs   += accel_macs();
}

/*---------------------------------------------------------------------------
 * nn_conv2d_array -- convolution on the MAC array, one window at a time
 *-------------------------------------------------------------------------*
 *
 * For each output pixel we gather its receptive field (in_ch * kh * kw
 * values) into a contiguous vector, push it as activations, and run each
 * output channel's weights against it. That makes K = in_ch*kh*kw, which is
 * a long reduction -- exactly what the array wants, and exactly what dot4
 * could not get at because of the non-contiguous window layout.
 *
 * The gather buffer is static, not on the stack: 4 KB of stack is all we
 * have (see link.ld), and a large local array here would silently overflow
 * it and corrupt whatever sits below.
 */
#define MAX_PATCH 512   /* in_ch * kh * kw for our models; checked below */

static int8_t patch_buf[MAX_PATCH];

void nn_conv2d_array(const int8_t *in, int32_t in_h, int32_t in_w,
                     const nn_conv_t *layer, int8_t *out,
                     int32_t *out_h, int32_t *out_w, int32_t *scratch)
{
    (void)scratch;

    const int32_t kh = layer->kh, kw = layer->kw;
    const int32_t stride = layer->stride, pad = layer->pad;
    const int32_t in_ch = layer->in_ch, out_ch = layer->out_ch;
    const int32_t patch_len = in_ch * kh * kw;

    const int32_t oh = (in_h + 2 * pad - kh) / stride + 1;
    const int32_t ow = (in_w + 2 * pad - kw) / stride + 1;

    if (patch_len > MAX_PATCH) {
        /* Fail loudly and predictably rather than smashing memory. Producing
         * zeros is obviously wrong downstream, which is the point. */
        g_accel_timeouts++;
        for (int32_t i = 0; i < out_ch * oh * ow; i++) out[i] = 0;
        *out_h = oh; *out_w = ow;
        return;
    }

    accel_clear_perf();

    for (int32_t oy = 0; oy < oh; oy++) {
        for (int32_t ox = 0; ox < ow; ox++) {

            /* ---- im2col for ONE output position ---------------------- */
            int32_t p = 0;
            for (int32_t ic = 0; ic < in_ch; ic++) {
                const int8_t *x_ic = in + (int32_t)ic * in_h * in_w;
                for (int32_t ky = 0; ky < kh; ky++) {
                    const int32_t iy = oy * stride + ky - pad;
                    for (int32_t kx = 0; kx < kw; kx++) {
                        const int32_t ix = ox * stride + kx - pad;
                        /* Outside the image -> zero. Exact for symmetric
                         * quantization; see the note in nn_fc_array. */
                        patch_buf[p++] = (iy < 0 || iy >= in_h ||
                                          ix < 0 || ix >= in_w)
                                         ? 0 : x_ic[iy * in_w + ix];
                    }
                }
            }

            int32_t i = 0;
            for (; i + 4 <= patch_len; i += 4) {
                accel_push_activation(pack4(patch_buf[i], patch_buf[i+1],
                                            patch_buf[i+2], patch_buf[i+3]));
            }
            if (i < patch_len) {
                int8_t t[4] = {0, 0, 0, 0};
                for (int32_t j = 0; i + j < patch_len; j++) t[j] = patch_buf[i + j];
                accel_push_activation(pack4(t[0], t[1], t[2], t[3]));
            }

            for (int32_t oc = 0; oc < out_ch; oc++) {
                const int8_t *w = layer->weights + (int32_t)oc * patch_len;

                int32_t k = 0;
                for (; k + 4 <= patch_len; k += 4) {
                    accel_push_weight(pack4(w[k], w[k+1], w[k+2], w[k+3]));
                }
                if (k < patch_len) {
                    int8_t t[4] = {0, 0, 0, 0};
                    for (int32_t j = 0; k + j < patch_len; j++) t[j] = w[k + j];
                    accel_push_weight(pack4(t[0], t[1], t[2], t[3]));
                }

                accel_set_dims(1, 1, (uint32_t)((patch_len + 3) / 4));
                accel_start();

                int32_t acc;
                if (accel_wait_done(ACCEL_POLL_LIMIT) != 0) {
                    g_accel_timeouts++;
                    acc = 0;
                } else {
                    acc = accel_pop_result();
                }
                acc += layer->bias ? layer->bias[oc] : 0;

                out[(oc * oh + oy) * ow + ox] =
                    (int8_t)nn_requantize(acc, layer->qp.multiplier,
                                          layer->qp.shift,
                                          layer->qp.zero_point,
                                          layer->qp.precision);
            }
        }
    }

    g_accel_total_cycles += accel_cycles();
    g_accel_total_stalls += accel_stalls();
    g_accel_total_macs   += accel_macs();

    *out_h = oh;
    *out_w = ow;
}
