/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2025-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

double *pU, *pV, *pA;

// This test checks if buffers are passed in the correct order to the task
// it is based on https://github.com/starpu-runtime/starpu/issues/78

void scal_cpu_func(void* buffers[], void* _args)
{
	(void) _args;
	double* A = (double*)STARPU_VARIABLE_GET_PTR(buffers[0]);
	double* U = (double*)STARPU_VARIABLE_GET_PTR(buffers[1]);
	double* V = (double*)STARPU_VARIABLE_GET_PTR(buffers[2]);
	int nthreads = starpu_combined_worker_get_size();
	FPRINTF(stderr, "nthreads = %d\nA = %p, U = %p, V = %p\n", nthreads, A, U, V);
	STARPU_ASSERT_MSG(U == pU, "Pointers are different %p %p\n", U, pU);
	STARPU_ASSERT_MSG(V == pV, "Pointers are different %p %p\n", V, pV);
	STARPU_ASSERT_MSG(A == pA, "Pointers are different %p %p\n", A, pA);
}

static struct starpu_codelet cl =
{
	.modes = {STARPU_R, STARPU_R, STARPU_R},
	.type = STARPU_FORKJOIN,
	.max_parallelism = INT_MAX,
	.cpu_funcs = {scal_cpu_func},
	.cpu_funcs_name = {"scal_cpu_func"},
	.nbuffers = 3
};

int main()
{
	int ret;

	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.nmpi_sc = 0;
	conf.ntcpip_sc = 0;
	conf.ncpus = 2;
	conf.sched_policy_name = "peager";
	conf.calibrate = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	pU = malloc(sizeof(double));
	pV = malloc(sizeof(double));
	pA = malloc(sizeof(double));

	starpu_data_handle_t U;
	starpu_data_handle_t V;
	starpu_data_handle_t A;
	starpu_variable_data_register(&U, STARPU_MAIN_RAM, (uintptr_t)pU, sizeof(double));
	starpu_variable_data_register(&V, STARPU_MAIN_RAM, (uintptr_t)pV, sizeof(double));
	starpu_variable_data_register(&A, STARPU_MAIN_RAM, (uintptr_t)pA, sizeof(double));

	FPRINTF(stderr, "A = %p, U = %p, V = %p\n", pA, pU, pV);
	ret = starpu_task_insert(&cl,
				 STARPU_R, A,
				 STARPU_R, U,
				 STARPU_R, V,
				 0);
	if (ret == -ENODEV)
		ret = STARPU_TEST_SKIPPED;
	else
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	starpu_data_unregister(A);
	starpu_data_unregister(U);
	starpu_data_unregister(V);

	free(pU);
	free(pV);
	free(pA);
	starpu_shutdown();
	STARPU_RETURN(ret);
}
