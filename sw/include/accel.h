/*===========================================================================
 * accel.h -- Intrinsics for the custom instructions and the MAC array
 *===========================================================================
 *
 * This header is how C code reaches the hardware we added. It has two halves:
 *
 *   1. Inline-assembly intrinsics for the DOT4 / DOT4A / ACCRD custom
 *      instructions (implemented in rtl/dot4_pcpi.v).
 *   2. Memory-mapped register access for the MAC array (rtl/accel_top.v).
 *
 * You should never need to hand-assemble anything or remember an address.
 *===========================================================================*/

#ifndef ACCEL_H
#define ACCEL_H

#include <stdint.h>

/*---------------------------------------------------------------------------
 * Part 1: custom instruction intrinsics
 *-------------------------------------------------------------------------*/
/*
 * HOW `.insn r` WORKS
 * -------------------
 * The assembler does not know our instruction mnemonics -- we invented them.
 * But GNU as provides `.insn`, which builds an instruction word from its
 * fields directly:
 *
 *     .insn r opcode, funct3, funct7, rd, rs1, rs2
 *
 * So `.insn r 0x0B, 0, 0, %0, %1, %2` emits an R-type instruction with
 * opcode 0x0B (custom-0), funct3 0, funct7 0 -- which rtl/dot4_pcpi.v decodes
 * as DOT4. The %0/%1/%2 are filled in by the compiler with whatever registers
 * it chose for the C variables; we never pick registers ourselves, which is
 * what makes this safe to use anywhere.
 *
 * WHY `volatile`
 * --------------
 * DOT4A and ACCRD have a side effect the compiler cannot see: they modify a
 * hardware accumulator. Without `volatile`, the optimiser is entitled to
 * delete a DOT4A whose result is unused, or hoist it out of a loop, or
 * reorder it past an ACCRD. All three would be silently wrong. `volatile`
 * tells GCC "this has effects you do not know about -- keep it exactly where
 * and as often as I wrote it".
 */

/* Four-lane signed dot product: returns sum of a[i]*b[i] for i in 0..3,
 * where each 32-bit word holds four packed int8 values (lane 0 = low byte). */
static inline int32_t dot4(uint32_t a, uint32_t b)
{
    int32_t rd;
    __asm__ volatile (".insn r 0x0B, 0, 0, %0, %1, %2"
                      : "=r"(rd) : "r"(a), "r"(b));
    return rd;
}

/* Accumulate a four-lane dot product into the hardware accumulator.
 * Does not return anything -- read the total back with accrd(). */
static inline void dot4a(uint32_t a, uint32_t b)
{
    __asm__ volatile (".insn r 0x0B, 0, 1, x0, %0, %1"
                      : : "r"(a), "r"(b) : "memory");
}

/* Read the hardware accumulator AND clear it, atomically in one instruction.
 * Doing it as two instructions would be a race if an interrupt landed in
 * between. */
static inline int32_t accrd(void)
{
    int32_t rd;
    __asm__ volatile (".insn r 0x0B, 0, 2, %0, x0, x0"
                      : "=r"(rd) : : "memory");
    return rd;
}

/* Helper: pack four int8 values into one 32-bit word in lane order.
 * The casts to uint8_t before shifting matter -- shifting a negative signed
 * char left is undefined behaviour and, worse, sign-extends into the
 * neighbouring lanes and corrupts them. */
static inline uint32_t pack4(int8_t l0, int8_t l1, int8_t l2, int8_t l3)
{
    return ((uint32_t)(uint8_t)l0)        |
           ((uint32_t)(uint8_t)l1) <<  8  |
           ((uint32_t)(uint8_t)l2) << 16  |
           ((uint32_t)(uint8_t)l3) << 24;
}

/* Load four consecutive int8 values as one packed word.
 *
 * NOTE: this does an unaligned-safe byte-by-byte load rather than casting the
 * pointer to uint32_t*. PicoRV32 is built with CATCH_MISALIGN=1, so a
 * misaligned 32-bit load TRAPS. Weight arrays are not guaranteed to be
 * 4-byte aligned at every offset a convolution walks to, so the pointer cast
 * would work in testing and then trap on real data. */
static inline uint32_t load4(const int8_t *p)
{
    return pack4(p[0], p[1], p[2], p[3]);
}

/*---------------------------------------------------------------------------
 * Part 2: the memory-mapped MAC array
 *-------------------------------------------------------------------------*/
/*
 * These addresses MUST match rtl/accel_pkg.vh. A mismatch here is a classic
 * silent bug: writing the length into the start register makes the
 * accelerator either do nothing or run forever.
 */
#define ACCEL_BASE        0x40000000u

#define ACCEL_REG_CTRL    (*(volatile uint32_t *)(ACCEL_BASE + 0x00))
#define ACCEL_REG_STATUS  (*(volatile uint32_t *)(ACCEL_BASE + 0x04))
#define ACCEL_REG_M       (*(volatile uint32_t *)(ACCEL_BASE + 0x08))
#define ACCEL_REG_N       (*(volatile uint32_t *)(ACCEL_BASE + 0x0C))
#define ACCEL_REG_K       (*(volatile uint32_t *)(ACCEL_BASE + 0x10))
#define ACCEL_REG_WBUF    (*(volatile uint32_t *)(ACCEL_BASE + 0x40))
#define ACCEL_REG_ABUF    (*(volatile uint32_t *)(ACCEL_BASE + 0x44))
#define ACCEL_REG_RESULT  (*(volatile uint32_t *)(ACCEL_BASE + 0x48))
#define ACCEL_REG_CYCLES  (*(volatile uint32_t *)(ACCEL_BASE + 0x50))
#define ACCEL_REG_STALLS  (*(volatile uint32_t *)(ACCEL_BASE + 0x54))
#define ACCEL_REG_MACS    (*(volatile uint32_t *)(ACCEL_BASE + 0x58))
#define ACCEL_REG_CONFIG  (*(volatile uint32_t *)(ACCEL_BASE + 0x5C))

/* CTRL bits */
#define ACCEL_CTRL_START      0x1u
#define ACCEL_CTRL_SOFT_RESET 0x2u
#define ACCEL_CTRL_CLEAR_PERF 0x4u

/* STATUS bits */
#define ACCEL_STATUS_BUSY     0x1u
#define ACCEL_STATUS_DONE     0x2u
#define ACCEL_STATUS_HAS_RESULT 0x4u

/* Every one of those macros dereferences a `volatile` pointer. Without
 * volatile the compiler would see `while (ACCEL_REG_STATUS & BUSY);` as a
 * loop whose condition never changes, hoist the load out, and spin forever
 * on a stale value. This is the single most common bare-metal bug there is. */

static inline void accel_set_dims(uint32_t m, uint32_t n, uint32_t k)
{
    ACCEL_REG_M = m;
    ACCEL_REG_N = n;
    ACCEL_REG_K = k;
}

static inline void accel_push_weight(uint32_t packed) { ACCEL_REG_WBUF = packed; }
static inline void accel_push_activation(uint32_t packed) { ACCEL_REG_ABUF = packed; }

static inline void accel_start(void) { ACCEL_REG_CTRL = ACCEL_CTRL_START; }

/* Block until the current job finishes.
 *
 * The bounded loop is deliberate. An unbounded `while (busy);` turns any
 * hardware bug into a silent hang with no output at all -- the single most
 * confusing failure mode there is. Returning an error instead lets main.c
 * print something useful. */
static inline int accel_wait_done(uint32_t max_polls)
{
    for (uint32_t i = 0; i < max_polls; i++) {
        if (ACCEL_REG_STATUS & ACCEL_STATUS_DONE) return 0;
    }
    return -1;   /* timed out -- the caller should report this loudly */
}

static inline int32_t accel_pop_result(void) { return (int32_t)ACCEL_REG_RESULT; }

/* Performance counters. These are the evidence for RQ3: `stalls` counts
 * cycles where the array was ready but had no data, separately from cycles of
 * real work. */
static inline void accel_clear_perf(void) { ACCEL_REG_CTRL = ACCEL_CTRL_CLEAR_PERF; }
static inline uint32_t accel_cycles(void) { return ACCEL_REG_CYCLES; }
static inline uint32_t accel_stalls(void) { return ACCEL_REG_STALLS; }
static inline uint32_t accel_macs(void)   { return ACCEL_REG_MACS; }

/* Discover the geometry that was actually synthesised.
 *
 * The sweep builds many array shapes from one firmware image, so the software
 * must ADAPT rather than assume. Hardcoding 4x4 here would silently compute
 * the wrong answer on every other configuration -- the worst kind of bug,
 * because it still produces a plausible number. */
static inline uint32_t accel_array_h(void) { return  ACCEL_REG_CONFIG        & 0xFF; }
static inline uint32_t accel_array_w(void) { return (ACCEL_REG_CONFIG >>  8) & 0xFF; }
static inline uint32_t accel_precision(void) { return (ACCEL_REG_CONFIG >> 16) & 0xFF; }

/* How many DISTINCT outputs one pass can produce.
 *
 * accel_top.v fans a 32-bit buffer word out to the array as lane (j % 4), so
 * columns 4..7 of an 8-wide array see the same four lanes as columns 0..3.
 * Only min(ARRAY_W, 4) columns are therefore independent. This is a real
 * limitation of the 32-bit operand path, documented in docs/REVIEW.md, and it
 * means an 8-wide build does NOT double throughput under this mapping. */
static inline uint32_t accel_lanes(void)
{
    const uint32_t w = accel_array_w();
    return (w < 4u) ? w : 4u;
}

#endif /* ACCEL_H */
