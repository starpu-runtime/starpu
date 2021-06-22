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

#include "bench_helper.h"


int comp_double(const void*_a, const void*_b)
{
	const double* a = _a;
	const double* b = _b;

	if(*a < *b)
		return -1;
	else if(*a > *b)
		return 1;
	else
		return 0;
}


uint64_t bench_next_size(uint64_t len)
{
	uint64_t next = len * MULT_DEFAULT;

	if(next <= len)
		next++;

	return next;
}


uint64_t bench_nb_iterations(int iterations, uint64_t len)
{
	const uint64_t max_data = NX_MAX;

	if(len == 0)
		len = 1;

	uint64_t data_size = ((uint64_t)iterations * (uint64_t)len);

	if(data_size  > max_data)
	{
		iterations = (max_data / (uint64_t)len);
		if(iterations < 2)
			iterations = 2;
	}

	return iterations;
}
