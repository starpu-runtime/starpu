/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <stdlib.h>
#include <stdio.h>
#include <starpu_scheduler.h>
#include "../helper.h"

/* This examples shows the case of determining statically whether data is in CPU
 * memory or DFE memory, and using the basic Maxeler interface */

#include "MyTasks.h"
#include <MaxSLiCInterface.h>
#define SIZE (192/sizeof(int32_t))

void fpga_impl(void *buffers[], void *cl_arg)
{
	(void)cl_arg;

	int32_t *ptrA = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[0]);
	int32_t *ptrB = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[1]);
	int32_t *ptrC = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[2]);

	int size = STARPU_VECTOR_GET_NX(buffers[0]);

	// XXX: would rather use a scratch buffer
	size_t ptrCT1 = 0x00000000000000c0;

	size_t ptrAT2 = ptrCT1;
	size_t ptrBT2 = ptrCT1;
	size_t ptrCT2 = 0x0000000000000180;

	size_t ptrAT3 = ptrCT2;
	size_t ptrBT3 = ptrCT2;

	printf("Loading DFE memory.\n");

	/* C = A+B */
	MyTasks_interfaceT1(size, ptrCT1, ptrA, ptrB);
	printf("T1 finished\n");

	/* C = A*B */
	MyTasks_interfaceT2(size, ptrAT2, ptrBT2, ptrCT2);
	printf("T2 finished\n");

	/* C = A+B */
	MyTasks_interfaceT3(size, ptrAT3, ptrBT3, ptrC);
	printf("T3 finished\n");

	printf("Running DFE.\n");
}

static struct starpu_codelet cl =
{
	.max_fpga_funcs = {fpga_impl},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_CPU, STARPU_SPECIFIC_NODE_CPU, STARPU_SPECIFIC_NODE_CPU},
};

void fpga_impl1(void *buffers[], void *cl_arg)
{
	(void)cl_arg;

	int32_t *ptrA = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[0]);
	int32_t *ptrB = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[1]);
	size_t	 ptrC = (size_t)   STARPU_VECTOR_GET_PTR(buffers[2]); /* FPGA */

	int size = STARPU_VECTOR_GET_NX(buffers[0]);

	printf("T1 with %p %p %zu\n", ptrA, ptrB, ptrC);
	/* C = A+B */
	MyTasks_interfaceT1(size, ptrC, ptrA, ptrB);
	printf("T1 finished\n");
}

static struct starpu_codelet cl1 =
{
	.max_fpga_funcs = {fpga_impl1},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_CPU, STARPU_SPECIFIC_NODE_CPU, STARPU_SPECIFIC_NODE_LOCAL},
};

void fpga_impl2(void *buffers[], void *cl_arg)
{
	(void)cl_arg;

	size_t ptrA = (size_t) STARPU_VECTOR_GET_PTR(buffers[0]); /* FPGA */
	size_t ptrB = (size_t) STARPU_VECTOR_GET_PTR(buffers[1]); /* FPGA */
	size_t ptrC = (size_t) STARPU_VECTOR_GET_PTR(buffers[2]); /* FPGA */

	int size = STARPU_VECTOR_GET_NX(buffers[0]);

	printf("T2 with %zu %zu %zu\n", ptrA, ptrB, ptrC);
	/* C = A*B */
	MyTasks_interfaceT2(size, ptrA, ptrB, ptrC);
	printf("T2 finished\n");
}

static struct starpu_codelet cl2 =
{
	.max_fpga_funcs = {fpga_impl2},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W}
	/* local by default */
};

void fpga_impl3(void *buffers[], void *cl_arg)
{
	(void)cl_arg;

	size_t	 ptrA = (size_t)   STARPU_VECTOR_GET_PTR(buffers[0]); /* FPGA */
	size_t	 ptrB = (size_t)   STARPU_VECTOR_GET_PTR(buffers[1]); /* FPGA */
	int32_t *ptrC = (int32_t*) STARPU_VECTOR_GET_PTR(buffers[2]);

	int size = STARPU_VECTOR_GET_NX(buffers[0]);

	printf("T3 with %zu %zu %p\n", ptrA, ptrB, ptrC);
	/* C = A+B */
	MyTasks_interfaceT3(size, ptrA, ptrB, ptrC);
	printf("T3 finished\n");
}

static struct starpu_codelet cl3 =
{
	.max_fpga_funcs = {fpga_impl3},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_W},
	.specific_nodes = 1,
	.nodes = {STARPU_SPECIFIC_NODE_LOCAL, STARPU_SPECIFIC_NODE_LOCAL, STARPU_SPECIFIC_NODE_CPU},
};

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	starpu_data_handle_t handle_a, handle_b, handle_ct1, handle_ct2, handle_c;
	int ret;

	starpu_conf_init(&conf);
	conf.sched_policy_name = "eager";
	conf.calibrate = 0;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Enable profiling */
	starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

	int32_t a[SIZE];
	int32_t b[SIZE];
	int32_t c[SIZE];

	int i;
	for(i = 0; i < SIZE; ++i)
	{
		a[i] = random() % 100;
		b[i] = random() % 100;
	}

	starpu_vector_data_register(&handle_a, STARPU_MAIN_RAM, (uintptr_t) &a, SIZE, sizeof(a[0]));
	starpu_vector_data_register(&handle_b, STARPU_MAIN_RAM, (uintptr_t) &b, SIZE, sizeof(b[0]));

	starpu_vector_data_register(&handle_ct1, -1, 0, SIZE, sizeof(c[0]));
	starpu_vector_data_register(&handle_ct2, -1, 0, SIZE, sizeof(c[0]));

	starpu_vector_data_register(&handle_c, STARPU_MAIN_RAM, (uintptr_t) &c, SIZE, sizeof(c[0]));

#if 0
	ret = starpu_task_insert(&cl, STARPU_R, handle_a, STARPU_R, handle_b, STARPU_W, handle_c, STARPU_TASK_SYNCHRONOUS, 1, 0);
	fprintf(stderr,"task submitted %d\n", ret);
#else
	ret = starpu_task_insert(&cl1, STARPU_R, handle_a, STARPU_R, handle_b, STARPU_W, handle_ct1, 0);
	fprintf(stderr,"task submitted %d\n", ret);
	ret = starpu_task_insert(&cl2, STARPU_R, handle_ct1, STARPU_R, handle_ct1, STARPU_W, handle_ct2, 0);
	fprintf(stderr,"task submitted %d\n", ret);
	ret = starpu_task_insert(&cl3, STARPU_R, handle_ct2, STARPU_R, handle_ct2, STARPU_W, handle_c, 0);
	fprintf(stderr,"task submitted %d\n", ret);
#endif

	starpu_data_unregister(handle_a);
	starpu_data_unregister(handle_b);
	starpu_data_unregister(handle_c);

	ret = EXIT_SUCCESS;

	for (i = 0; i < SIZE; ++i)
	{
		int ct1 = a[i] + b[i];
		int ct2 = ct1 * ct1;
		int ct3 = ct2 + ct2;

		if (c[i] != ct3)
			ret = EXIT_FAILURE;

		if (i < 10)
		{
			printf("%d == %d\n", c[i], ct3);
			if (c[i] != ct3)
				printf("OOOPS\n");
		}
	}

	starpu_shutdown();

	if (ret == EXIT_SUCCESS)
		printf("OK!\n");

	return ret;
}
