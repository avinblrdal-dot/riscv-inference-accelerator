/* Minimal probe: one small dot product on the MAC array, with the answer
 * computed in plain C alongside it. Isolates the accelerator contract from
 * the full model, so a failure points at one thing. */
#include <stdint.h>
#include "nn.h"
#include "accel.h"

#define UART_DATA   (*(volatile uint32_t *)0x10000000u)
#define UART_STATUS (*(volatile uint32_t *)0x10000004u)
static void putc_(char c){ while (UART_STATUS & 1u){} UART_DATA=(uint32_t)(uint8_t)c; }
static void puts_(const char*s){ while(*s) putc_(*s++); }
static void putd_(int32_t v){ char b[12]; int i=0; if(!v){putc_('0');return;}
  if(v<0){putc_('-');} while(v){int32_t d=v%10; if(d<0)d=-d; b[i++]=(char)('0'+d); v/=10;}
  while(i) putc_(b[--i]); }
static void kv(const char*k,int32_t v){ puts_(k); putc_('='); putd_(v); putc_('\n'); }

#define N 8
static int8_t A[N];
static int8_t W0[N], W1[N], W2[N], W3[N];

int main(void)
{
    int32_t i, ref[4], acc[4];

    puts_("PROBE start\n");
    kv("array_h", (int32_t)accel_array_h());
    kv("array_w", (int32_t)accel_array_w());
    kv("lanes",   (int32_t)accel_lanes());

    for (i = 0; i < N; i++) {
        A[i]  = (int8_t)(i + 1);
        W0[i] = (int8_t)1;
        W1[i] = (int8_t)2;
        W2[i] = (int8_t)(-1);
        W3[i] = (int8_t)3;
    }
    for (i = 0; i < 4; i++) { ref[i] = 0; acc[i] = 0; }
    for (i = 0; i < N; i++) {
        ref[0] += A[i]*W0[i]; ref[1] += A[i]*W1[i];
        ref[2] += A[i]*W2[i]; ref[3] += A[i]*W3[i];
    }
    kv("ref0", ref[0]); kv("ref1", ref[1]);
    kv("ref2", ref[2]); kv("ref3", ref[3]);

    /* --- drive the array by hand, exactly as nn_array.c intends --- */
    ACCEL_REG_CTRL = ACCEL_CTRL_SOFT_RESET;
    for (i = 0; i < N; i++) {
        accel_push_activation(pack4(A[i], 0, 0, 0));
        accel_push_weight(pack4(W0[i], W1[i], W2[i], W3[i]));
    }
    puts_("pushed\n");

    accel_set_dims(1u, 4u, (uint32_t)N);
    kv("m_readback", (int32_t)ACCEL_REG_M);
    kv("n_readback", (int32_t)ACCEL_REG_N);
    kv("k_readback", (int32_t)ACCEL_REG_K);

    accel_start();
    puts_("started\n");

    {
        int32_t polls = 0;
        uint32_t st = 0;
        for (polls = 0; polls < 200000; polls++) {
            st = ACCEL_REG_STATUS;
            if (st & ACCEL_STATUS_DONE) break;
        }
        kv("polls", polls);
        kv("status", (int32_t)st);
    }

    kv("cycles", (int32_t)accel_cycles());
    kv("macs",   (int32_t)accel_macs());
    kv("stalls", (int32_t)accel_stalls());

    for (i = 0; i < 16; i++) {
        int32_t v = accel_pop_result();
        puts_("cell["); putd_(i/4); putc_(','); putd_(i%4); puts_("]=");
        putd_(v); putc_('\n');
        if (i < 4) acc[i] = v;
    }
    kv("got0", acc[0]); kv("got1", acc[1]);
    kv("got2", acc[2]); kv("got3", acc[3]);

    puts_((acc[0]==ref[0] && acc[1]==ref[1] &&
           acc[2]==ref[2] && acc[3]==ref[3]) ? "PROBE PASS\n" : "PROBE FAIL\n");

    *(volatile uint32_t *)0x30000000u = 0;
    return 0;
}
