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

#include "increment_codelet.h"

void cpu_increment(void *descr[], void *arg)
{
	(void)arg;
	unsigned *var = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	(*var)++;
}

struct starpu_codelet increment_codelet =
{
	.cpu_funcs = {cpu_increment},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {cuda_host_increment},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
	// TODO
	//.opencl_funcs = {dummy_func},
	.cpu_funcs_name = {"cpu_increment"},
	.model = NULL,
	.modes = { STARPU_RW },
	.nbuffers = 1
};

