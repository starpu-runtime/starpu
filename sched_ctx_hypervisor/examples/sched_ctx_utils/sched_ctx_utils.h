#include <limits.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

#define START_BENCH 0
#define END_BENCH 1

void parse_args_ctx(int argc, char **argv);
void update_sched_ctx_timing_results(double gflops, double timing);
void construct_contexts(void (*bench)(float *mat, unsigned size, unsigned nblocks));
void end_contexts(void);
void start_2benchs(void (*bench)(float *mat, unsigned size, unsigned nblocks));
void start_1stbench(void (*bench)(float *mat, unsigned size, unsigned nblocks));
void start_2ndbench(void (*bench)(float *mat, unsigned size, unsigned nblocks));
