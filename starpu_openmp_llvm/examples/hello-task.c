/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2022 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

int array[] = {1, 2, 3, 4};

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

int main()
{
	int res=0;
#pragma omp parallel
#pragma omp master
	{
		FPRINTF(stderr, "Hello from %i\n", omp_get_thread_num());
#pragma omp task
		{
			sleep(2);
			FPRINTF(stderr, "Hey there\n");
		}
		for (int i = 0; i < 4; i++)
		{
#pragma omp task depend(in: array[i]) depend(inout: array[(i+1)%4])
			{
				array[(i+1)%4] = array[i];
				FPRINTF(stderr, "Hey %i\n", i);
			}
		}
	}
	FPRINTF(stderr, "array: ");
	for (int i = 0; i < 4; i++)
	{
		FPRINTF(stderr, "%i, ", array[i]);
		if (array[i] != 1)
		{
			FPRINTF(stderr, "\n");
			FPRINTF(stderr, "Incorrect value. Should be 1\n");
			res = 1;
		}
	}
	FPRINTF(stderr, "\n");
	return res;
}
