/*
 * main.c — thread/NUMA/reporting harness for the x86-64-v3 asm kernel.
 * Build:  cc -O2 -march=x86-64-v3 -Wall -Wextra -pthread \
 *            main.c stress_kernel.S -lnuma -o stress
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <numa.h>
#include <immintrin.h>

#include "stress.h"

#define FLOATS          16
#define INTS            8
#define ITERS_PER_BLOCK (1ULL << 30)   /* ~0.7 s/block on a 14900K P-core,
                                          ~2-3 s on an E-core. Tune freely. */
#define FLOP_PER_ITER   128.0          /* 8 FMA x 8 lanes x 2 */

_Static_assert(offsetof(stress_out, iadd)   == 256, "asm layout");
_Static_assert(offsetof(stress_out, ixor)   == 288, "asm layout");
_Static_assert(offsetof(stress_out, isub)   == 320, "asm layout");
_Static_assert(offsetof(stress_out, scalar) == 352, "asm layout");

static void fill_random(float *f, int32_t *d, unsigned *seed)
{
    for (int i = 0; i < FLOATS; i++)   /* [1,100]: keeps FP chains away
                                          from zero/denormal territory  */
        f[i] = 1.0f + 99.0f * (float)rand_r(seed) / (float)RAND_MAX;
    for (int i = 0; i < INTS; i++)
        d[i] = rand_r(seed) % 1000;
}

static void print_f8(const char *label, const float *a)
{
    printf("  %s: [", label);
    for (int i = 0; i < 8; i++)
        printf("%.4g%s", (double)a[i], i == 7 ? "" : ", ");
    printf("]\n");
}

static void print_i8(const char *label, const int32_t *a)
{
    printf("  %s: [", label);
    for (int i = 0; i < 8; i++)
        printf("%d%s", a[i], i == 7 ? "" : ", ");
    printf("]\n");
}

static void *worker(void *arg)
{
    long id = (long)arg;

    /* thread i -> CPU i. Simple and correct on hybrid (8P+16E) parts;
       the old phys/logical pairing math breaks on SMT-less E-cores.   */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((int)id, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);

    int node = (int)(id % (numa_max_node() + 1));
    numa_run_on_node(node);

    /* FTZ + DAZ: no denormal microcode assists inside the FMA chains. */
    _mm_setcsr(_mm_getcsr() | 0x8040);

    float   *f = numa_alloc_onnode(FLOATS * sizeof *f, node); /* page-,
                                                  hence 32B-aligned */
    int32_t *d = numa_alloc_onnode(INTS * sizeof *d, node);
    if (!f || !d) {
        fprintf(stderr, "thread %ld: numa_alloc failed\n", id);
        return NULL;
    }

    unsigned seed = (unsigned)time(NULL) ^ ((unsigned)id * 2654435761u);
    stress_out out;
    struct timespec t0, t1;

    for (unsigned long long block = 1; ; block++) {
        fill_random(f, d, &seed);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        stress_block(&out, f, d, ITERS_PER_BLOCK);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double dt = (double)(t1.tv_sec - t0.tv_sec)
                  + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
        double gflops = FLOP_PER_ITER * (double)ITERS_PER_BLOCK / dt * 1e-9;

        printf("\n[T%02ld cpu%02d node%d blk%llu] %.1f GFLOP/s\n",
               id, sched_getcpu(), node, block, gflops);
        print_f8("FMA acc0", out.fp[0]);
        print_i8("ADD acc ", out.iadd);
        print_i8("XOR acc ", out.ixor);
    }

    /* not reached */
    numa_free(f, FLOATS * sizeof *f);
    numa_free(d, INTS * sizeof *d);
    return NULL;
}

int main(void)
{
    if (numa_available() < 0) {
        fprintf(stderr, "NUMA not supported on this system!\n");
        return 1;
    }
    int n = get_nprocs();

    printf("x86-64-v3 stress: %d threads, %llu iters/block, %.0f FLOP/iter\n",
           n, (unsigned long long)ITERS_PER_BLOCK, FLOP_PER_ITER);

    pthread_t *t = malloc((size_t)n * sizeof *t);
    if (!t)
        return EXIT_FAILURE;

    for (long i = 0; i < n; i++) {
        if (pthread_create(&t[i], NULL, worker, (void *)i)) {
            fprintf(stderr, "pthread_create failed (%ld)\n", i);
            free(t);
            return EXIT_FAILURE;
        }
    }
    for (int i = 0; i < n; i++)
        pthread_join(t[i], NULL);

    free(t);
    return 0;
}
