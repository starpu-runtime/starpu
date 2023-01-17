/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <math.h>
#include <starpu_mpi.h>
#include "helper.h"

#define NX_MIN 1

#ifdef STARPU_QUICK_CHECK
	#define MULT_DEFAULT 4
	#define LOOPS_DEFAULT 50
	#define NX_MAX (64 * 1024) // kB
#elif !defined(STARPU_LONG_CHECK)
	#define MULT_DEFAULT 4
	#define LOOPS_DEFAULT 10000
	#define NX_MAX (128 * 1024 * 1024) // kB
#else
	#define MULT_DEFAULT 2
	#define LOOPS_DEFAULT 100000
	#define NX_MAX (512 * 1024 * 1024) // kB
#endif

int comp_double(const void*_a, const void*_b);
uint64_t bench_next_size(uint64_t len);
uint64_t bench_nb_iterations(int iterations, uint64_t len);
