/*===========================================================================
 * perf.h -- Cycle counting from inside C
 *===========================================================================
 *
 * This is how research question RQ1 gets answered. RQ1 asks: what fraction of
 * cycles does an unmodified RISC-V core spend inside MAC work? That fraction
 * sets the Amdahl ceiling -- the best speedup any accelerator could ever
 * achieve, no matter how good it is:
 *
 *     max_speedup = 1 / (1 - fraction_accelerated)
 *
 * If MACs are 80% of cycles, no accelerator beats 5x overall. Knowing that
 * number early stops the team chasing an impossible target, and it is a
 * genuinely interesting result to report on its own.
 *
 * HOW IT WORKS
 * ------------
 * rv32i defines a `rdcycle` instruction that reads a free-running 64-bit
 * cycle counter. We enabled it in soc_top.v with ENABLE_COUNTERS=1 and read
 * only the low 32 bits, which is ample: 2^32 cycles at 100 MHz is ~43
 * seconds, and an inference takes milliseconds.
 *
 * MEASUREMENT DISCIPLINE
 * ----------------------
 * Reading the counter is not free (a few cycles). For short regions that
 * overhead matters, so measure a REPEATED region and divide, rather than
 * timing one iteration. perf_region_t below does the bookkeeping.
 *===========================================================================*/

#ifndef PERF_H
#define PERF_H

#include <stdint.h>

/* Read the low 32 bits of the cycle counter. */
static inline uint32_t perf_cycles(void)
{
    uint32_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
}

/* Read the retired-instruction counter. Cycles/instructions gives IPC, which
 * is how you tell "the core is stalled on memory" apart from "the core is
 * executing a lot of instructions". */
static inline uint32_t perf_instret(void)
{
    uint32_t c;
    __asm__ volatile ("rdinstret %0" : "=r"(c));
    return c;
}

/* Also available as memory-mapped counters in soc_top.v, which work even in
 * builds where the CSR instructions are disabled. */
#define PERF_MMIO_CYCLES (*(volatile uint32_t *)0x20000000u)
#define PERF_MMIO_INSTR  (*(volatile uint32_t *)0x20000004u)

/* Accumulates time spent in a named region across many entries. */
typedef struct {
    const char *name;
    uint32_t    total_cycles;
    uint32_t    entries;
    uint32_t    t0;        /* internal: timestamp at the last begin */
} perf_region_t;

static inline void perf_begin(perf_region_t *r) { r->t0 = perf_cycles(); }

static inline void perf_end(perf_region_t *r)
{
    r->total_cycles += perf_cycles() - r->t0;
    r->entries++;
}

/* Subtract the cost of the measurement itself, so that timing a very short
 * region does not report mostly its own overhead. Call once at startup and
 * subtract the result per measured region. */
static inline uint32_t perf_overhead(void)
{
    const uint32_t a = perf_cycles();
    const uint32_t b = perf_cycles();
    return b - a;
}

#endif /* PERF_H */
