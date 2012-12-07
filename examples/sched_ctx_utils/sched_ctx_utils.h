#include <limits.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

void parse_args_ctx(int argc, char **argv);
void update_sched_ctx_timing_results(double gflops, double timing);
void construct_contexts(void (*bench)(unsigned size, unsigned nblocks));
void start_2benchs(void (*bench)(unsigned size, unsigned nblocks));
void start_1stbench(void (*bench)(unsigned size, unsigned nblocks));
void start_2ndbench(void (*bench)(unsigned size, unsigned nblocks));
