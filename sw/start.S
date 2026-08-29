/*===========================================================================
 * start.S -- Bare-metal startup code
 *===========================================================================
 *
 * This is the very first thing the CPU executes. PicoRV32 is configured with
 * PROGADDR_RESET = 0, so it begins at address 0, and link.ld places
 * .text.init there -- which is this file.
 *
 * WHY ASSEMBLY AND NOT C:
 *   C requires a working stack pointer and zeroed globals before main() can
 *   run. Neither exists at reset. Something has to set them up in a language
 *   that does not itself need them, and that language is assembly.
 *
 * WHAT IT DOES, in order:
 *   1. Point the stack pointer at the top of RAM.
 *   2. Set up the global pointer (gp) for small-data addressing.
 *   3. Zero the .bss section, because C guarantees uninitialised globals
 *      are zero and nothing else will do it.
 *   4. Call main().
 *   5. If main ever returns, stop cleanly rather than running off into
 *      whatever bytes follow.
 *
 * Note there is no .data copy loop. On a flash-based system .data would have
 * to be copied from flash into RAM; here $readmemh preloads one unified RAM,
 * so it is already in place. See the comment in link.ld.
 *===========================================================================*/

    .section .text.init
    .global _start
    .global _exit

_start:
    /* ---- 1. Stack pointer -------------------------------------------- */
    /* __stack_top comes from link.ld. la is a pseudo-instruction that the
     * assembler expands into the right lui/addi pair. */
    la      sp, __stack_top

    /* ---- 2. Global pointer ------------------------------------------- */
    /* gp lets the compiler reach small globals with one instruction instead
     * of two. The .option norelax around it is REQUIRED and subtle: without
     * it, the linker's relaxation pass would rewrite this very instruction
     * to be relative to gp -- which is not set up yet. The result is a gp
     * pointing at garbage and globals silently reading the wrong memory. */
.option push
.option norelax
    la      gp, __global_pointer$
.option pop

    /* ---- 3. Zero .bss ------------------------------------------------ */
    la      t0, __bss_start
    la      t1, __bss_end
bss_loop:
    bge     t0, t1, bss_done
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       bss_loop
bss_done:

    /* ---- 4. Into C --------------------------------------------------- */
    /* argc = 0, argv = NULL: main() ignores them, but the ABI says a0/a1
     * hold them and leaving them as garbage is asking for trouble. */
    li      a0, 0
    li      a1, 0
    call    main

    /* ---- 5. Clean stop ----------------------------------------------- */
    /* main returned. Its exit code is already in a0; hand it to _exit. */
_exit:
    /* Writing to the simulation exit port (0x3000_0000) makes soc_top call
     * $finish, which ends the simulation with our exit code. On real
     * hardware that address is unmapped and simply does nothing, so we fall
     * through to the halt loop below -- correct behaviour in both worlds. */
    li      t0, 0x30000000
    sw      a0, 0(t0)

halt:
    /* Spin forever. On the FPGA this is where the program ends up after
     * inference; a real deployment would execute WFI here instead so the
     * clock can gate and the board drops to microamps. That change is worth
     * making before any energy measurement -- see docs/MEASUREMENT_PROTOCOL.md. */
    j       halt
