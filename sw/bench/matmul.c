/*===========================================================================
 * matmul.c -- Plain matrix-multiply benchmark: where does the speedup stop?
 *===========================================================================
 *
 * WHY THIS EXISTS
 * ---------------
 * Suggested by Dr. Bhatia (Sept 8 meeting): instead of only running neural
 * networks, run large matrix multiplications and find "the size of data
 * chunk at the point where [the speedup] starts decreasing". A matmul is the
 * cleanest possible workload for that question -- no im2col, no ReLU, no
 * requantization, just C = A x B -- so every cycle measured here is either
 * multiply-accumulate work or the cost of feeding the hardware.
 *
 * It also does something the two neural-network workloads cannot. Those run
 * at batch size 1 (one sensor window at a time), so only ROW 0 of the MAC
 * array ever does useful work (see nn_array.c, nn_fc_array). A matmul has
 * many rows of A, so here the array's rows are used too: on a 4x4 build one
 * pass computes a 4x4 block of C instead of a 1x4 strip.
 *
 * WHAT IS MEASURED
 * ----------------
 *   C[M][N] = A[M][K] * B[K][N]        int8 inputs, int32 outputs
 *
 * B is stored TRANSPOSED (Bt[N][K]) so that column j of B is a contiguous
 * row, exactly like a fully-connected layer's weight matrix. That keeps the
 * array kernel's packing identical to nn_array.c's proven operand contract.
 *
 * Three experiments, each printed as one "mm ..." line per measurement:
 *
 *   ksweep  M = N = 16, K = 8 .. 1024. K is the length of each dot product,
 *           i.e. the size of the data chunk streamed through the array per
 *           output. This is the professor's question.
 *   chunk   M = N = 16, K = 256, the hardware fed in chunks of 4 .. 256
 *           values per pass (forced, no weight reuse). Isolates the cost of
 *           splitting a long dot product into short pieces.
 *   square  M = N = K = 4 .. 64. The "how big a matrix" view.
 *
 * Two firmware images, selected by -D flags in sw/Makefile, as main.c does:
 *   BENCH_CPU    software baseline (rv32i, every multiply is a libgcc call)
 *                and the DOT4 custom instruction. Neither touches the MAC
 *                array, so this image is run once, not once per array shape.
 *   BENCH_ARRAY  the memory-mapped MAC array. Run on every array shape and
 *                buffer depth.
 *
 * CORRECTNESS
 * -----------
 * Every measurement prints a checksum of the whole C matrix. The runner
 * (sweep/run_matmul_sweep.py) recomputes C in Python from the same
 * pseudo-random inputs and rejects any row whose checksum differs. A fast
 * wrong answer is never recorded as fast -- the same rule as the main sweep.
 *
 * Only the kernel call is timed. Filling the matrices and computing the
 * checksum happen outside the timed region.
 *===========================================================================*/

#include <stdint.h>
#include "perf.h"
#include "accel.h"

/*---------------------------------------------------------------------------
 * Minimal UART output (same as main.c)
 *-------------------------------------------------------------------------*/
#define UART_DATA   (*(volatile uint32_t *)0x10000000u)
#define UART_STATUS (*(volatile uint32_t *)0x10000004u)

static void uart_putc(char c)
{
    while (UART_STATUS & 1u) { }
    UART_DATA = (uint32_t)(uint8_t)c;
}

static void uart_puts(const char *s)
{
    while (*s) uart_putc(*s++);
}

static void uart_putd(int32_t v)
{
    char buf[12];
    int  i = 0;
    int  neg = (v < 0);

    if (v == 0) { uart_putc('0'); return; }
    while (v != 0) {
        int32_t digit = v % 10;
        if (digit < 0) digit = -digit;
        buf[i++] = (char)('0' + digit);
        v /= 10;
    }
    if (neg) uart_putc('-');
    while (i > 0) uart_putc(buf[--i]);
}

/* " key=value", no newline -- one measurement is one line of these. */
static void kv(const char *key, int32_t value)
{
    uart_putc(' ');
    uart_puts(key);
    uart_putc('=');
    uart_putd(value);
}

static void ks(const char *key, const char *value)
{
    uart_putc(' ');
    uart_puts(key);
    uart_putc('=');
    uart_puts(value);
}

/*---------------------------------------------------------------------------
 * Operands
 *-------------------------------------------------------------------------*/
/* 16 KB each. Sized for the largest cases: 16 x 1024 (ksweep) and 64 x 64
 * (square). C is 64 x 64 int32 = 16 KB. Static, not stack: link.ld gives
 * 4 KB of stack. Total ~48 KB of the 64 KB RAM -- link.ld's ASSERT fails the
 * build if the code pushes it over. */
#define MAX_OPERAND_BYTES 16384
#define MAX_C_ELEMS       4096

static int8_t  mat_a[MAX_OPERAND_BYTES];   /* A [M][K], row-major           */
static int8_t  mat_bt[MAX_OPERAND_BYTES];  /* Bt[N][K], i.e. B transposed   */
static int32_t mat_c[MAX_C_ELEMS];         /* C [M][N], row-major           */

/* xorshift32: no multiply, so filling 32 KB costs little even on rv32i.
 * sweep/run_matmul_sweep.py reproduces this generator bit for bit. Each
 * experiment uses a PREFIX of the same fixed stream: A[i][k] is
 * mat_a[i*K + k] for whatever K that experiment uses. */
#define MATMUL_SEED 0x2545F491u

static void fill_operands(void)
{
    uint32_t x = MATMUL_SEED;
    int32_t i;
    for (i = 0; i < MAX_OPERAND_BYTES; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        mat_a[i] = (int8_t)(x >> 24);
    }
    for (i = 0; i < MAX_OPERAND_BYTES; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        mat_bt[i] = (int8_t)(x >> 24);
    }
}

/* h = h*31 + c over C in row-major order, mod 2^32. Untimed. */
static int32_t checksum(int32_t m, int32_t n)
{
    uint32_t h = 0;
    int32_t i;
    for (i = 0; i < m * n; i++) h = h * 31u + (uint32_t)mat_c[i];
    return (int32_t)h;
}

/*---------------------------------------------------------------------------
 * Kernels
 *-------------------------------------------------------------------------*/
#if defined(BENCH_CPU)

/* The baseline: what an unmodified rv32i core does. Optimised at -O2 like
 * every other baseline in this project. */
static void mm_baseline(int32_t m, int32_t n, int32_t k)
{
    int32_t i, j, kk;
    for (i = 0; i < m; i++) {
        const int8_t *a = mat_a + i * k;
        for (j = 0; j < n; j++) {
            const int8_t *b = mat_bt + j * k;
            int32_t acc = 0;
            for (kk = 0; kk < k; kk++) acc += (int32_t)a[kk] * (int32_t)b[kk];
            mat_c[i * n + j] = acc;
        }
    }
}

/* DOT4: four multiply-adds per custom instruction, accumulated in the
 * coprocessor. Requires k % 4 == 0, which every size used here satisfies. */
static void mm_dot4(int32_t m, int32_t n, int32_t k)
{
    int32_t i, j, kk;
    for (i = 0; i < m; i++) {
        const int8_t *a = mat_a + i * k;
        for (j = 0; j < n; j++) {
            const int8_t *b = mat_bt + j * k;
            (void)accrd();   /* clear the accumulator */
            for (kk = 0; kk < k; kk += 4) dot4a(load4(a + kk), load4(b + kk));
            mat_c[i * n + j] = accrd();
        }
    }
}

#endif /* BENCH_CPU */

#if defined(BENCH_ARRAY)

#define ACCEL_POLL_LIMIT 4000000u
static int32_t g_timeouts;

/* Pop one whole tile (ARRAY_H x ARRAY_W, row-major) and add the part we
 * asked for -- rows [0, rows), columns [0, lanes) -- into acc[r*4 + c]. The
 * rest must still be popped or it would be misread as the next tile. Nested
 * counters rather than e / W, because rv32i has no divide instruction. */
static void drain_tile(int32_t rows, int32_t lanes, int32_t *acc)
{
    const int32_t h = (int32_t)accel_array_h();
    const int32_t w = (int32_t)accel_array_w();
    int32_t r, c;
    for (r = 0; r < h; r++) {
        for (c = 0; c < w; c++) {
            const int32_t v = accel_pop_result();
            if (r < rows && c < lanes) acc[r * 4 + c] += v;
        }
    }
}

static void push_weights(const int8_t *const *b, int32_t lanes,
                         int32_t k0, int32_t klen)
{
    int32_t kk;
    for (kk = k0; kk < k0 + klen; kk++) {
        accel_push_weight(pack4(
            (lanes > 0) ? b[0][kk] : 0,
            (lanes > 1) ? b[1][kk] : 0,
            (lanes > 2) ? b[2][kk] : 0,
            (lanes > 3) ? b[3][kk] : 0));
    }
}

/* Activation word at k holds ONE value per ROW: lane r feeds array row r.
 * Same contract as weights-per-column -- see the header of nn_array.c. */
static void push_activations(const int8_t *const *a, int32_t rows,
                             int32_t k0, int32_t klen)
{
    int32_t kk;
    for (kk = k0; kk < k0 + klen; kk++) {
        accel_push_activation(pack4(
            (rows > 0) ? a[0][kk] : 0,
            (rows > 1) ? a[1][kk] : 0,
            (rows > 2) ? a[2][kk] : 0,
            (rows > 3) ? a[3][kk] : 0));
    }
}

static void run_pass(int32_t rows, int32_t lanes, int32_t klen, int32_t *acc)
{
    accel_set_dims((uint32_t)rows, (uint32_t)lanes, (uint32_t)klen);
    accel_start();
    if (accel_wait_done(ACCEL_POLL_LIMIT) != 0) { g_timeouts++; return; }
    drain_tile(rows, lanes, acc);
}

/* C = A x B on the array.
 *
 * resident = 1: if a column group's whole K-length weight block fits in the
 *   buffer, load it ONCE and stream every row group of A past it (weight
 *   reuse, as nn_conv2d_array does). Falls back to chunked if it does not fit.
 * resident = 0: always chunked -- each pass rewinds both buffers, pushes
 *   `chunk` values of weights AND activations, runs, and the partial sums
 *   are added in software. No reuse.
 *
 * Returns 1 if the resident path was actually used. */
static int32_t mm_array(int32_t m, int32_t n, int32_t k,
                        int32_t chunk, int32_t resident)
{
    const int32_t rows_max  = (accel_array_h() < 4u) ? (int32_t)accel_array_h() : 4;
    const int32_t lanes_max = (int32_t)accel_lanes();
    const int32_t wcap = (int32_t)accel_wbuf_words();
    const int32_t acap = (int32_t)accel_abuf_words();
    const int32_t cap  = (wcap < acap) ? wcap : acap;
    const int32_t use_resident = resident && (k <= cap);
    int32_t i0, j0, r, c, k0;

    if (chunk > cap) chunk = cap;
    if (chunk > k)   chunk = k;

    for (j0 = 0; j0 < n; j0 += lanes_max) {
        const int8_t *b[4];
        int32_t lanes = n - j0;
        if (lanes > lanes_max) lanes = lanes_max;
        for (c = 0; c < 4; c++)
            b[c] = mat_bt + ((c < lanes) ? (j0 + c) : j0) * k;

        if (use_resident) {
            ACCEL_REG_CTRL = ACCEL_CTRL_RST_WBUF;
            push_weights(b, lanes, 0, k);
        }

        for (i0 = 0; i0 < m; i0 += rows_max) {
            const int8_t *a[4];
            int32_t acc[16];
            int32_t rows = m - i0;
            if (rows > rows_max) rows = rows_max;
            for (r = 0; r < 4; r++)
                a[r] = mat_a + ((r < rows) ? (i0 + r) : i0) * k;
            for (r = 0; r < 16; r++) acc[r] = 0;

            if (use_resident) {
                ACCEL_REG_CTRL = ACCEL_CTRL_RST_ABUF;
                push_activations(a, rows, 0, k);
                run_pass(rows, lanes, k, acc);
            } else {
                for (k0 = 0; k0 < k; k0 += chunk) {
                    int32_t klen = k - k0;
                    if (klen > chunk) klen = chunk;
                    ACCEL_REG_CTRL = ACCEL_CTRL_SOFT_RESET;
                    push_weights(b, lanes, k0, klen);
                    push_activations(a, rows, k0, klen);
                    run_pass(rows, lanes, klen, acc);
                }
            }

            for (r = 0; r < rows; r++)
                for (c = 0; c < lanes; c++)
                    mat_c[(i0 + r) * n + j0 + c] = acc[r * 4 + c];
        }
    }
    return use_resident;
}

#endif /* BENCH_ARRAY */

/*---------------------------------------------------------------------------
 * Driver
 *-------------------------------------------------------------------------*/
static void line_head(const char *exp, const char *impl,
                      int32_t m, int32_t n, int32_t k)
{
    uart_puts("mm");
    ks("exp", exp);
    ks("impl", impl);
    kv("m", m);
    kv("n", n);
    kv("k", k);
}

static void line_tail(uint32_t cycles, int32_t m, int32_t n)
{
    kv("cycles", (int32_t)cycles);
    kv("checksum", checksum(m, n));
    uart_putc('\n');
}

#if defined(BENCH_CPU)
static void measure_cpu(const char *exp, int32_t m, int32_t n, int32_t k)
{
    uint32_t t0, cyc;

    t0 = perf_cycles();
    mm_baseline(m, n, k);
    cyc = perf_cycles() - t0;
    line_head(exp, "baseline", m, n, k);
    line_tail(cyc, m, n);

    t0 = perf_cycles();
    mm_dot4(m, n, k);
    cyc = perf_cycles() - t0;
    line_head(exp, "dot4", m, n, k);
    line_tail(cyc, m, n);
}
#endif

#if defined(BENCH_ARRAY)
static void measure_array(const char *exp, int32_t m, int32_t n, int32_t k,
                          int32_t chunk, int32_t resident)
{
    uint32_t t0, cyc;
    int32_t used, cap, eff_chunk;

    g_timeouts = 0;
    t0 = perf_cycles();
    used = mm_array(m, n, k, chunk, resident);
    cyc = perf_cycles() - t0;

    cap = (int32_t)((accel_wbuf_words() < accel_abuf_words())
                    ? accel_wbuf_words() : accel_abuf_words());
    eff_chunk = used ? k : chunk;
    if (eff_chunk > cap) eff_chunk = cap;
    if (eff_chunk > k)   eff_chunk = k;

    line_head(exp, "array", m, n, k);
    ks("mode", used ? "resident" : "chunked");
    kv("chunk", eff_chunk);
    kv("timeouts", g_timeouts);
    line_tail(cyc, m, n);
}
#endif

static const int32_t ksweep_k[] = { 8, 16, 32, 64, 128, 256, 512, 1024 };
#if defined(BENCH_ARRAY)
static const int32_t chunk_sizes[] = { 4, 8, 16, 32, 64, 128, 256 };
#endif
static const int32_t square_n[] = { 4, 8, 16, 32, 64 };

#define COUNT(a) ((int32_t)(sizeof(a) / sizeof((a)[0])))

int main(void)
{
    int32_t i;

    uart_puts("\n=== matmul benchmark ===\n");
    fill_operands();

#if defined(BENCH_CPU)
    uart_puts("image=cpu\n");
    for (i = 0; i < COUNT(ksweep_k); i++) measure_cpu("ksweep", 16, 16, ksweep_k[i]);
    for (i = 0; i < COUNT(square_n); i++)
        measure_cpu("square", square_n[i], square_n[i], square_n[i]);
#endif

#if defined(BENCH_ARRAY)
    uart_puts("image=array\n");
    {
        uart_puts("array_h=");  uart_putd((int32_t)accel_array_h());
        uart_puts(" array_w="); uart_putd((int32_t)accel_array_w());
        uart_puts(" wbuf=");    uart_putd((int32_t)accel_wbuf_words());
        uart_puts(" abuf=");    uart_putd((int32_t)accel_abuf_words());
        uart_putc('\n');
    }
    /* ksweep: the best the hardware can do at each K -- resident when the
     * block fits, otherwise chunks as large as the buffer allows. */
    for (i = 0; i < COUNT(ksweep_k); i++)
        measure_array("ksweep", 16, 16, ksweep_k[i], ksweep_k[i], 1);

    /* chunk: forced chunking at every chunk size the buffer can hold. */
    {
        const int32_t cap = (int32_t)((accel_wbuf_words() < accel_abuf_words())
                                      ? accel_wbuf_words() : accel_abuf_words());
        for (i = 0; i < COUNT(chunk_sizes); i++)
            if (chunk_sizes[i] <= cap)
                measure_array("chunk", 16, 16, 256, chunk_sizes[i], 0);
    }

    for (i = 0; i < COUNT(square_n); i++)
        measure_array("square", square_n[i], square_n[i], square_n[i],
                      square_n[i], 1);
#endif

    uart_puts("=== done ===\n");
    return 0;
}
