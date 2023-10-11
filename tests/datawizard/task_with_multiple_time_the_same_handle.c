/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * Test passing the same handle several times to the same task
 */

void sum_cpu(void * descr[], void *cl_arg)
{
	(void)cl_arg;
	double * v_dst = (double *) STARPU_VECTOR_GET_PTR(descr[0]);
	double * v_src = (double *) STARPU_VECTOR_GET_PTR(descr[1]);
	STARPU_ASSERT(v_dst == v_src);
	*v_dst+=*v_src;
}

void sum3_cpu(void * descr[], void *cl_arg)
{
	(void)cl_arg;
	double * v_src1 = (double *) STARPU_VECTOR_GET_PTR(descr[0]);
	double * v_src2 = (double *) STARPU_VECTOR_GET_PTR(descr[1]);
	double * v_dst = (double *) STARPU_VECTOR_GET_PTR(descr[2]);
	STARPU_ASSERT(v_dst == v_src1);
	STARPU_ASSERT(v_dst == v_src2);
	*v_dst+=*v_src1+*v_src2;
}

void sum4_cpu(void * descr[], void *cl_arg)
{
	(void)cl_arg;
	double * v_src1 = (double *) STARPU_VECTOR_GET_PTR(descr[0]);
	double * v_src2 = (double *) STARPU_VECTOR_GET_PTR(descr[1]);
	double * v_dst1 = (double *) STARPU_VECTOR_GET_PTR(descr[2]);
	double * v_dst2 = (double *) STARPU_VECTOR_GET_PTR(descr[3]);
	STARPU_ASSERT(v_src1 == v_dst1);
	STARPU_ASSERT(v_src2 == v_dst2);
	*v_dst2 = (*v_dst1+=*v_src1+*v_src2);
}

static struct starpu_codelet sum_cl =
{
	.cpu_funcs = {sum_cpu},
	.cpu_funcs_name = {"sum_cpu"},
	.nbuffers = 2,
	.modes={STARPU_RW,STARPU_R}
};

static struct starpu_codelet sum3_cl =
{
	.cpu_funcs = {sum3_cpu},
	.cpu_funcs_name = {"sum3_cpu"},
	.nbuffers = 3,
	.modes={STARPU_R,STARPU_R,STARPU_RW}
};

static struct starpu_codelet sum4_cl =
{
	.cpu_funcs = {sum4_cpu},
	.cpu_funcs_name = {"sum4_cpu"},
	.nbuffers = 4,
	.modes={STARPU_R,STARPU_R,STARPU_RW,STARPU_RW}
};

#define N 10
int main(void)
{
	starpu_data_handle_t handle;
	int ret = 0;
	double value[N] = { 1.0 };
	int i;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	conf.ncpus = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret=starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;

	starpu_vector_data_register(&handle,0,(uintptr_t)&value,N,sizeof(double));

	for (i=0; i<2; i++)
	{
		ret = starpu_task_insert(&sum_cl,
		                   STARPU_RW, handle,
		                   STARPU_R, handle,
		                   0);
		if (ret == -ENODEV) goto enodev;
		ret = starpu_task_insert(&sum3_cl,
		                   STARPU_R, handle,
		                   STARPU_R, handle,
		                   STARPU_RW, handle,
		                   0);
		if (ret == -ENODEV) goto enodev;
	}

	starpu_data_acquire(handle, STARPU_R);
	if (value[0] != 36)
	{
		FPRINTF(stderr, "value is %f instead of %f\n", value[0], 36.);
		ret = EXIT_FAILURE;
	}
	starpu_data_release(handle);

	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = 2,
	};
	starpu_data_partition(handle, &f);

	starpu_task_insert(&sum4_cl,
			STARPU_R,starpu_data_get_sub_data(handle,1,0),
			STARPU_R,starpu_data_get_sub_data(handle,1,1),
			STARPU_RW,starpu_data_get_sub_data(handle,1,0),
			STARPU_RW,starpu_data_get_sub_data(handle,1,1),
			0);

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);

	starpu_task_wait_for_all();
	starpu_data_unregister(handle);

	starpu_shutdown();
	return ret;

enodev:
	starpu_data_unregister(handle);
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
