#ifndef STRESS_H
#define STRESS_H

#include <stdint.h>

/* Layout consumed by stress_kernel.S — do NOT reorder. */
typedef struct {
    float    fp[8][8];   /* +0   8x FMA accumulator chains */
    int32_t  iadd[8];    /* +256 vpaddd chain              */
    int32_t  ixor[8];    /* +288 vpxor  chain              */
    int32_t  isub[8];    /* +320 vpsubd chain              */
    uint64_t scalar;     /* +352 BMI2 rorx chain           */
} __attribute__((aligned(32))) stress_out;

/* f: >=16 floats, 32B aligned; d: >=8 int32, 32B aligned */
void stress_block(stress_out *out, const float *f,
                  const int32_t *d, uint64_t iters);

#endif
