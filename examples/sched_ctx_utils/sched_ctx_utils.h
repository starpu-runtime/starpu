/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <limits.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

void parse_args_ctx(int argc, char **argv);
void update_sched_ctx_timing_results(double gflops, double timing);
void construct_contexts();
void start_2benchs(void (*bench)(unsigned size, unsigned nblocks));
void start_1stbench(void (*bench)(unsigned size, unsigned nblocks));
void start_2ndbench(void (*bench)(unsigned size, unsigned nblocks));
