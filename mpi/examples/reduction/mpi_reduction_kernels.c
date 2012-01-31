/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Centre National de la Recherche Scientifique
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

#include <starpu.h>

/*
 *	Codelet to create a neutral element
 */
void init_cpu_func(void *descr[], void *cl_arg)
{
	int *dot = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*dot = 0.0f;
}

/*
 *	Codelet to perform the reduction of two elements
 */
void redux_cpu_func(void *descr[], void *cl_arg)
{
	int *dota = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *dotb = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);

	*dota = *dota + *dotb;
}

/*
 *	Dot product codelet
 */
void dot_cpu_func(void *descr[], void *cl_arg)
{
	int *local_x = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	int *local_y = (int *)STARPU_VARIABLE_GET_PTR(descr[1]);
	int *dot = (int *)STARPU_VARIABLE_GET_PTR(descr[2]);

	*dot += *local_x + *local_y;
}

