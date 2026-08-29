/*===========================================================================
 * link.ld -- Linker script for the bare-metal RISC-V build
 *===========================================================================
 *
 * WHAT A LINKER SCRIPT IS, for a beginner:
 *
 *   On a laptop, the operating system decides where your program lives in
 *   memory. Here there IS no operating system. Something has to say "code
 *   goes at address 0, variables go after it, the stack starts at the top".
 *   That something is this file.
 *
 *   Get it wrong and the failure is spectacular but uninformative: the CPU
 *   executes data, or the stack grows down into your variables and silently
 *   corrupts them.
 *
 * THE MEMORY LAYOUT MUST MATCH rtl/soc_top.v:
 *   soc_top has MEM_WORDS 32-bit words of RAM starting at address 0, and
 *   PROGADDR_RESET = 0, so the CPU begins executing at address 0. Therefore
 *   .text must start at 0 and the first thing there must be start.S.
 *
 *   64 KB here = MEM_WORDS 16384 in soc_top.v. If you change one, change the
 *   other, or the linker will happily produce a binary that does not fit.
 *===========================================================================*/

MEMORY
{
    /* rwx because it is one unified RAM holding code and data alike. */
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 64K
}

/* The stack grows DOWNWARD from the top of RAM. 4 KB is generous for this
 * workload: the inference code uses static buffers rather than recursion or
 * large locals, precisely so that stack depth stays predictable. */
__stack_size = 4K;

ENTRY(_start)

SECTIONS
{
    /* ---- Code -------------------------------------------------------- */
    .text : {
        /* .text.init MUST come first: address 0 is where the CPU starts,
         * and start.S puts _start in .text.init to guarantee it lands there.
         * If some other object file's code got placed first, the CPU would
         * begin executing in the middle of an unrelated function. */
        *(.text.init)
        *(.text)
        *(.text.*)
        . = ALIGN(4);
    } > RAM

    /* ---- Read-only data ---------------------------------------------- */
    /* Quantized weights live here. They are const, so they stay in .rodata
     * and are never copied to RAM -- which matters, because the weights are
     * the largest thing in the image. */
    .rodata : {
        *(.rodata)
        *(.rodata.*)
        . = ALIGN(4);
    } > RAM

    /* ---- Initialised data -------------------------------------------- */
    /* On a system that boots from flash, .data would have to be COPIED from
     * flash to RAM at startup. Here code and data share one RAM that is
     * preloaded by $readmemh, so no copy is needed -- start.S deliberately
     * does not do one. Say so out loud, because "why is there no data copy
     * in the startup code" is a reasonable question to ask. */
    .data : {
        *(.data)
        *(.data.*)
        *(.sdata)
        *(.sdata.*)
        . = ALIGN(4);
    } > RAM

    /* ---- Zero-initialised data --------------------------------------- */
    /* .bss holds variables that start at zero. They occupy no space in the
     * image; start.S zeroes the region at boot. These symbols tell it where. */
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > RAM

    /* ---- Heap and stack ---------------------------------------------- */
    __heap_start = .;
    __stack_top  = ORIGIN(RAM) + LENGTH(RAM);

    /* Fail the BUILD if code plus data plus stack does not fit. Far better
     * than discovering it as a mysterious runtime corruption. */
    ASSERT(__bss_end + __stack_size <= __stack_top,
           "ERROR: program does not fit in RAM -- shrink the model, or raise LENGTH here and MEM_WORDS in rtl/soc_top.v")

    /* Discard debug/exception sections we cannot use on bare metal. */
    /DISCARD/ : {
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.comment)
        *(.riscv.attributes)
    }
}
