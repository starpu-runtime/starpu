/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "multiformat_types.h"
#include "../test_interfaces.h"
#include "../../../helper.h"

static void test_multiformat_cpu_func(void *buffers[], void *args);
#ifdef STARPU_USE_CUDA
extern void test_multiformat_cuda_func(void *buffers[], void *args);
#endif
#ifdef STARPU_USE_OPENCL
extern void test_multiformat_opencl_func(void *buffers[], void *args);
#endif
void test_multiformat_mic_func(void *buffers[], void *args);

static struct point array_of_structs[N_ELEMENTS];
static struct point array_of_structs_dummy[N_ELEMENTS];

static starpu_data_handle_t multiformat_handle;
static starpu_data_handle_t multiformat_dummy_handle;

struct test_config multiformat_config =
{
	.cpu_func      = test_multiformat_cpu_func,
#ifdef STARPU_USE_CUDA
	.cuda_func     = test_multiformat_cuda_func,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func   = test_multiformat_opencl_func,
#endif
	.cpu_func_name = "test_multiformat_mic_func",
	.handle        = &multiformat_handle,
	.dummy_handle  = &multiformat_dummy_handle,
	.copy_failed   = SUCCESS,
	.name          = "multiformat_interface"
};

static void
test_multiformat_cpu_func(void *buffers[], void *args)
{
	STARPU_SKIP_IF_VALGRIND;

	struct point *aos;
	int n, i;
	int factor;

	aos = (struct point *) STARPU_MULTIFORMAT_GET_CPU_PTR(buffers[0]);
	n = STARPU_MULTIFORMAT_GET_NX(buffers[0]);
	factor = *(int *) args;

	for (i = 0; i < n; i++)
	{
		FPRINTF(stderr, "(%d %d) [%d]", aos[i].x, aos[i].y, factor);
		if (aos[i].x != i * factor || aos[i].y != i * factor)
		{
			multiformat_config.copy_failed = FAILURE;
		}
		aos[i].x = -aos[i].x;
		aos[i].y = -aos[i].y;
	}
	FPRINTF(stderr, "\n");
}

void test_multiformat_mic_func(void *buffers[], void *args)
{
	STARPU_SKIP_IF_VALGRIND;

	printf("MIC\n");

	struct struct_of_arrays *soa;
	int n, i;
	int factor;

	soa = (struct struct_of_arrays *) STARPU_MULTIFORMAT_GET_MIC_PTR(buffers[0]);
	n = STARPU_MULTIFORMAT_GET_NX(buffers[0]);
	factor = *(int *) args;

	for (i = 0; i < n; i++)
	{
		FPRINTF(stderr, "(%d %d) [%d]", soa->x[i], soa->y[i], factor);
		if (soa->x[i] != i * factor || soa->y[i] != i * factor)
		{
			multiformat_config.copy_failed = 1;
		}
		soa->x[i] = -soa->x[i];
		soa->y[i] = -soa->y[i];
	}
	FPRINTF(stderr, "\n");
}

#ifdef STARPU_USE_CUDA
extern struct starpu_codelet cpu_to_cuda_cl;
extern struct starpu_codelet cuda_to_cpu_cl;
#endif

#ifdef STARPU_USE_OPENCL
extern struct starpu_codelet cpu_to_opencl_cl;
extern struct starpu_codelet opencl_to_cpu_cl;
#endif

#ifdef STARPU_USE_MIC
extern struct starpu_codelet cpu_to_mic_cl;
extern struct starpu_codelet mic_to_cpu_cl;
#endif

struct starpu_multiformat_data_interface_ops format_ops =
{
#ifdef STARPU_USE_CUDA
	.cuda_elemsize = 2* sizeof(float),
	.cpu_to_cuda_cl = &cpu_to_cuda_cl,
	.cuda_to_cpu_cl = &cuda_to_cpu_cl,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_elemsize = 2 * sizeof(float),
	.cpu_to_opencl_cl = &cpu_to_opencl_cl,
	.opencl_to_cpu_cl = &opencl_to_cpu_cl,
#endif
#ifdef STARPU_USE_MIC
	.mic_elemsize = 2 * sizeof(float),
	.cpu_to_mic_cl = &cpu_to_mic_cl,
	.mic_to_cpu_cl = &mic_to_cpu_cl,
#endif
	.cpu_elemsize = sizeof(struct point),
};

static void
register_data(void)
{
	int i;

	for (i = 0; i < N_ELEMENTS; i++)
	{
		array_of_structs[i].x = i;
		array_of_structs[i].y = i;
	}
	starpu_multiformat_data_register(&multiformat_handle,
					 STARPU_MAIN_RAM,
					 &array_of_structs,
					 N_ELEMENTS,
					 &format_ops);
	starpu_multiformat_data_register(&multiformat_dummy_handle,
					 STARPU_MAIN_RAM,
					 &array_of_structs_dummy,
					 N_ELEMENTS,
					 &format_ops);
}

static void
unregister_data(void)
{
	starpu_data_unregister(multiformat_handle);
	starpu_data_unregister(multiformat_dummy_handle);
}

int main(int argc, char **argv)
{
#ifdef STARPU_USE_CPU
	int ret;
	struct data_interface_test_summary summary;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.ncuda = 2;
	conf.nopencl = 1;
	conf.nmic = -1;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV || starpu_cpu_worker_get_count() == 0)
		return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	register_data();

	run_tests(&multiformat_config, &summary);

	data_interface_test_summary_print(stderr, &summary);

	unregister_data();

	starpu_shutdown();

	return data_interface_test_summary_success(&summary);
#else
	/* Without the CPU, there is no point in using the multiformat
	 * interface, so this test is pointless. */
	return STARPU_TEST_SKIPPED;
#endif
}
