/*
 * stress_kernel.S — x86-64-v3 (AVX2 + FMA3 + BMI2) stress kernel
 * System V AMD64 ABI, Linux.
 *
 * void stress_block(stress_out *out,        rdi  (32B aligned)
 *                   const float   *f,       rsi  (32B aligned, >= 16 floats)
 *                   const int32_t *d,       rdx  (32B aligned, >=  8 int32)
 *                   uint64_t       iters);  rcx
 *
 * Hot loop, per iteration:
 *   - 8x vfmadd213ps ymm   -> saturates both 256-bit FMA ports (p0/p1):
 *        8 FMA / 2 ports = 4 cycles, 8 * 8 lanes * 2 = 128 FLOP
 *        => 32 FLOP/cycle, the AVX2 ceiling of a Raptor Cove P-core
 *   - 3x integer SIMD (vpaddd/vpxor/vpsubd) -> scheduler steers to p5,
 *        no contention with the FMA ports (vector *shifts* would go p0/p1,
 *        so they are deliberately avoided)
 *   - 1x rorx (BMI2, flag-less) scalar side chain -> p0/p6 spare slots
 *   - sub/jnz macro-fused -> p6
 *   ~13 uops / 4 cycles: FMA-throughput bound, front-end has slack.
 *
 * FP chains compute acc = c*acc + b with c = 0.999999, b in [1,100]:
 * fixed point ~1e6*b, so values stay in normal range (no denormal
 * assists). Caller sets FTZ/DAZ as extra insurance.
 *
 * 8 independent accumulators because FMA latency(4) x ports(2) = 8
 * in-flight ops are needed for full throughput.
 */

    .intel_syntax noprefix
    .text
    .globl  stress_block
    .type   stress_block, @function
    .p2align 6
stress_block:
    test    rcx, rcx
    jz      .Lret_clean                 /* iters == 0: ymm never touched */

    /* ---- constants & sources (read-only inside the loop) ---------- */
    vbroadcastss ymm8,  dword ptr [rip + .Lc_decay]   /* c = 0.999999  */
    vmovaps      ymm9,  ymmword ptr [rsi]             /* b0 = f[0..7]  */
    vmovaps      ymm13, ymmword ptr [rsi + 32]        /* b1 = f[8..15] */
    vmovdqa      ymm12, ymmword ptr [rdx]             /* d  = int[0..7]*/

    /* ---- 8 independent FP accumulator chains ---------------------- */
    vmovaps ymm0, ymm9
    vmovaps ymm1, ymm13
    vaddps  ymm2, ymm9, ymm13
    vmulps  ymm3, ymm9, ymm13
    vmovaps ymm4, ymm9
    vmovaps ymm5, ymm13
    vaddps  ymm6, ymm9, ymm13
    vmulps  ymm7, ymm9, ymm13

    /* ---- integer accumulators ------------------------------------- */
    vmovdqa ymm10, ymm12                /* vpaddd chain                */
    vpshufd ymm11, ymm12, 0x1B          /* vpxor  chain                */
    vpxor   ymm14, ymm14, ymm14         /* vpsubd chain                */

    mov     r8, rcx                     /* BMI2 scalar chain seed      */

    .p2align 5
.Lloop:
    vfmadd213ps ymm0, ymm8, ymm9        /* acc = c*acc + b0            */
    vfmadd213ps ymm1, ymm8, ymm13
    vfmadd213ps ymm2, ymm8, ymm9
    vfmadd213ps ymm3, ymm8, ymm13
    vpaddd      ymm10, ymm10, ymm12
    vfmadd213ps ymm4, ymm8, ymm9
    vfmadd213ps ymm5, ymm8, ymm13
    vpxor       ymm11, ymm11, ymm10
    vfmadd213ps ymm6, ymm8, ymm9
    vfmadd213ps ymm7, ymm8, ymm13
    vpsubd      ymm14, ymm14, ymm11
    rorx        r8, r8, 13
    sub         rcx, 1
    jnz         .Lloop

    /* ---- results -> out (offsets must match stress_out, stress.h) - */
    vmovaps ymmword ptr [rdi +   0], ymm0
    vmovaps ymmword ptr [rdi +  32], ymm1
    vmovaps ymmword ptr [rdi +  64], ymm2
    vmovaps ymmword ptr [rdi +  96], ymm3
    vmovaps ymmword ptr [rdi + 128], ymm4
    vmovaps ymmword ptr [rdi + 160], ymm5
    vmovaps ymmword ptr [rdi + 192], ymm6
    vmovaps ymmword ptr [rdi + 224], ymm7
    vmovdqa ymmword ptr [rdi + 256], ymm10
    vmovdqa ymmword ptr [rdi + 288], ymm11
    vmovdqa ymmword ptr [rdi + 320], ymm14
    mov     qword ptr  [rdi + 352], r8
    vzeroupper                          /* avoid AVX->SSE penalty in caller */
.Lret_clean:
    ret
    .size   stress_block, . - stress_block

    .section .rodata
    .p2align 2
.Lc_decay:
    .float  0.999999

    .section .note.GNU-stack, "", @progbits
