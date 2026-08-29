/*===========================================================================
 * nn.h -- Quantized neural network types and integer arithmetic
 *===========================================================================
 *
 * This header defines the integer inference primitives shared by every
 * software variant (baseline, dot4, array). The arithmetic here MUST match
 * train/quant_ref.py and rtl/requantize.v bit for bit -- train/verify_parity.py
 * proves it on every golden vector.
 *
 * If you change anything in this file, you have changed the numerical
 * behaviour of the whole project. Re-run `make test` before committing, and
 * write down what changed in docs/DECISIONS.md.
 *
 * WHY EVERYTHING IS AN INTEGER
 * ----------------------------
 * The RISC-V core we use (PicoRV32, rv32i) has no floating-point unit. A
 * single float multiply would be emulated in software at a cost of dozens of
 * instructions, which would dominate the very MAC loop we are trying to
 * measure. More fundamentally, float results could not be reproduced
 * bit-exactly by the FPGA, so the parity harness -- the thing that lets us
 * distinguish a hardware bug from numerical drift -- would be impossible.
 *===========================================================================*/

#ifndef NN_H
#define NN_H

#include <stdint.h>

/* Baseline right shift for the fixed-point multiplier. Keep in sync with
 * M0_SHIFT in train/quant_ref.py and ACCEL_M0_SHIFT in rtl/accel_pkg.vh. */
#define NN_M0_SHIFT 31

/* Saturation limits as a function of precision (8 or 4). */
#define NN_QMIN(p) (-(1 << ((p) - 1)))
#define NN_QMAX(p) ((1 << ((p) - 1)) - 1)

/* Per-layer quantization parameters, produced offline by train/quantize.py
 * and emitted into sw/models/*.h. */
typedef struct {
    int32_t multiplier;  /* M0, normalized into [2^30, 2^31) */
    int32_t shift;       /* total right shift, = 31 + n      */
    int32_t zero_point;  /* 0 for the symmetric scheme we use */
    int32_t precision;   /* 8 or 4                            */
} nn_qparams_t;

/* A 2-D convolution layer with int8 weights and int32 biases. */
typedef struct {
    const int8_t  *weights;   /* [out_ch][in_ch][kh][kw], row-major */
    const int32_t *bias;      /* [out_ch], in accumulator units     */
    int32_t out_ch, in_ch, kh, kw, stride, pad;
    nn_qparams_t qp;
} nn_conv_t;

/* A fully-connected layer. */
typedef struct {
    const int8_t  *weights;   /* [out_dim][in_dim], row-major */
    const int32_t *bias;      /* [out_dim]                    */
    int32_t out_dim, in_dim;
    nn_qparams_t qp;
} nn_fc_t;

/*---------------------------------------------------------------------------
 * Core arithmetic (implemented in sw/src/quant.c)
 *-------------------------------------------------------------------------*/

/* Scale an int32 accumulator down to `precision` bits, integer-only.
 * Round half AWAY FROM ZERO, symmetrically -- see quant.c for why. */
int32_t nn_requantize(int32_t acc, int32_t multiplier, int32_t shift,
                      int32_t zero_point, int32_t precision);

/* Clamp to the representable range for the given precision. */
int32_t nn_saturate(int32_t v, int32_t precision);

/*---------------------------------------------------------------------------
 * Layer kernels
 *-------------------------------------------------------------------------*/

void nn_conv2d(const int8_t *in, int32_t in_h, int32_t in_w,
               const nn_conv_t *layer, int8_t *out,
               int32_t *out_h, int32_t *out_w, int32_t *scratch);

void nn_fc(const int8_t *in, const nn_fc_t *layer, int8_t *out);

void nn_relu(int8_t *x, int32_t n, int32_t zero_point);

void nn_maxpool2d(const int8_t *in, int32_t ch, int32_t h, int32_t w,
                  int32_t k, int8_t *out, int32_t *out_h, int32_t *out_w);

/* Index of the largest element -- the predicted class. */
int32_t nn_argmax(const int8_t *x, int32_t n);

#endif /* NN_H */
