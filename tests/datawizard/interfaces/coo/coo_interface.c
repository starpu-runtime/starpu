/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
#include "../test_interfaces.h"

#define NX 2
#define NY 2
#define MATRIX_SIZE (NX*NY)

#if defined(STARPU_USE_CPU) || defined(STAPRU_USE_MIC)
void test_coo_cpu_func(void *buffers[], void *args);
#endif
#ifdef STARPU_USE_CUDA
extern void test_coo_cuda_func(void *buffers[], void *args);
#endif
#ifdef STARPU_USE_OPENCL
extern void test_coo_opencl_func(void *buffers[], void *args);
#endif

static starpu_data_handle_t coo_handle, coo2_handle;

struct test_config coo_config =
{
#ifdef STARPU_USE_CPU
	.cpu_func      = test_coo_cpu_func,
#endif /* ! STARPU_USE_CPU */
#ifdef STARPU_USE_CUDA
	.cuda_func     = test_coo_cuda_func,
#endif /* !STARPU_USE_CUDA */
#ifdef STARPU_USE_OPENCL
	.opencl_func   = test_coo_opencl_func,
#endif /* !STARPU_USE_OPENCL */
#ifdef STARPU_USE_MIC
	.cpu_func_name = "test_coo_cpu_func",
#endif
	.handle        = &coo_handle,
	.dummy_handle  = &coo2_handle,
	.copy_failed   = SUCCESS,
	.name          = "coo_interface"
};

void
test_coo_cpu_func(void *buffers[], void *args)
{
	int factor = *(int *) args;
	int *values = (int *) STARPU_COO_GET_VALUES(buffers[0]);
	unsigned size = STARPU_COO_GET_NVALUES(buffers[0]);

	int i;
	for (i = 0; i < (int)size; i++)
	{
		if (values[i] != i * factor)
		{
			coo_config.copy_failed = FAILURE;
			return;
		}
		values[i] *= -1;
	}
}


static uint32_t columns[MATRIX_SIZE];
static uint32_t rows[MATRIX_SIZE];
static int values[MATRIX_SIZE];
static uint32_t columns2[MATRIX_SIZE];
static uint32_t rows2[MATRIX_SIZE];
static int values2[MATRIX_SIZE];

static void
register_data(void)
{
	/*
 	   We use the following matrix :

		+---+---+
		| 0 | 1 |
		+---+---+
		| 2 | 3 |
		+---+---+

	   Of course, we're not supposed to register the zeros, but it does not
	   matter for this test.
	 */

	columns[0] = 0;
	rows[0] = 0;
	values[0] = 0;

	columns[1] = 1;
	rows[1] = 0;
	values[1] = 1;

	columns[2] = 0;
	rows[2] = 1;
	values[2] = 2;

	columns[3] = 1;
	rows[3] = 1;
	values[3] = 3;


	int i;
	for (i = 0; i < MATRIX_SIZE; i++)
	{
		columns2[i] = -1;
		rows2[i] = -1;
		values2[i] = -1;
	}

	starpu_coo_data_register(&coo_handle,
				STARPU_MAIN_RAM,
				NX,
				NY,
				MATRIX_SIZE,
				columns,
				rows,
				(uintptr_t) values,
				sizeof(values[0]));
	starpu_coo_data_register(&coo2_handle,
				STARPU_MAIN_RAM,
				NX,
				NY,
				MATRIX_SIZE,
				columns2,
				rows2,
				(uintptr_t) values2,
				sizeof(values2[0]));
}

static void
unregister_data(void)
{
	starpu_data_unregister(coo_handle);
	starpu_data_unregister(coo2_handle);
}

int
main(int argc, char **argv)
{
	struct starpu_conf conf;
	struct data_interface_test_summary summary;

	starpu_conf_init(&conf);
	conf.ncuda = 2;
	conf.nopencl = 1;
	conf.nmic = -1;

	if (starpu_initialize(&conf, &argc, &argv) == -ENODEV || starpu_cpu_worker_get_count() == 0)
		goto enodev;

	register_data();

	run_tests(&coo_config, &summary);

	unregister_data();

	data_interface_test_summary_print(stderr, &summary);

	starpu_shutdown();
	return data_interface_test_summary_success(&summary);

enodev:
	return STARPU_TEST_SKIPPED;
}
