/* End-to-end host test: run the generated model in C and print every layer. */
#include <stdio.h>
#include "nn.h"
#include "model_weights.h"

static int8_t  buf_a[MODEL_MAX_TENSOR];
static int8_t  buf_b[MODEL_MAX_TENSOR];
static int32_t scratch[MODEL_MAX_TENSOR];

int main(void)
{
    int32_t h = MODEL_INPUT_H, w = MODEL_INPUT_W;
    int8_t *cur = buf_a, *nxt = buf_b;

    for (int32_t i = 0; i < MODEL_INPUT_CH*MODEL_INPUT_H*MODEL_INPUT_W; i++)
        cur[i] = model_test_input[i];

    for (int32_t li = 0; li < MODEL_NUM_LAYERS; li++) {
        const model_layer_t *L = &model_layers[li];
        int32_t oh, ow, n = 0;
        switch (L->kind) {
        case LAYER_CONV:
            nn_conv2d(cur, h, w, &L->conv, nxt, &oh, &ow, scratch);
            h = oh; w = ow; n = L->conv.out_ch*oh*ow;
            { int8_t *t = cur; cur = nxt; nxt = t; }
            break;
        case LAYER_RELU:
            nn_relu(cur, L->n_elems, 0); n = L->n_elems; break;
        case LAYER_MAXPOOL:
            nn_maxpool2d(cur, L->pool_ch, h, w, L->pool_k, nxt, &oh, &ow);
            h = oh; w = ow; n = L->pool_ch*oh*ow;
            { int8_t *t = cur; cur = nxt; nxt = t; }
            break;
        case LAYER_FC:
            nn_fc(cur, &L->fc, nxt); n = L->fc.out_dim;
            { int8_t *t = cur; cur = nxt; nxt = t; }
            break;
        }
        printf("LAYER %d n=%d:", li, n);
        for (int32_t i = 0; i < n; i++) printf(" %d", cur[i]);
        printf("\n");
    }
    printf("CLASS %d\n", nn_argmax(cur, MODEL_NUM_CLASSES));
    return 0;
}
