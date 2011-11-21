/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Institut National de Recherche en Informatique et Automatique
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
#ifndef TEST_INTERFACES_H
#define TEST_INTERFACES_H

typedef starpu_data_handle (*register_func_t) (void);

struct test_config {
	/* Returns a valid handle to a piece of data registered by StarPU */
	register_func_t register_func;

	/* StarPU codelets. The following functions should :
	 * 1) Check that the values are correct
	 * 2) Negate every element
	 */
	starpu_cpu_func_t cpu_func;
#ifdef STARPU_USE_CUDA
	starpu_cuda_func_t cuda_func;
#endif
#ifdef STARPU_USE_OPENCL
	starpu_opencl_func_t opencl_func;
#endif

	/* The previous codelets must update this field at the end of their
	 * execution. copy_failed must be 1 if the copy failed, 0 otherwise. */
	int copy_failed;

	/* A human-readable name for the test */
	const char *name;
};

#endif /* !TEST_INTERFACES_H */
