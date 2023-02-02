/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../../../helper.h"

#define NX 4
#define NY NX
#define NZ NX
#define NT NX

/* Prototypes */
static void register_data(void);
static void unregister_data(void);
void test_arr4d_cpu_func(void *buffers[], void *args);
#ifdef STARPU_USE_CUDA
extern void test_arr4d_cuda_func(void *buffers[], void *_args);
#endif
#ifdef STARPU_USE_OPENCL
extern void test_arr4d_opencl_func(void *buffers[], void *args);
#endif

static starpu_data_handle_t _arr4d_handle;
static starpu_data_handle_t _arr4d2_handle;

static int _arr4d[NX*NY*NZ*NT];
static int _arr4d2[NX*NY*NZ*NT];

struct test_config arr4d_config =
{
	.cpu_func      = test_arr4d_cpu_func,
#ifdef STARPU_USE_CUDA
	.cuda_func     = test_arr4d_cuda_func,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func   = test_arr4d_opencl_func,
#endif
	.handle        = &_arr4d_handle,
	.ptr           = _arr4d,
	.dummy_handle  = &_arr4d2_handle,
	.dummy_ptr     = _arr4d2,
	.copy_failed   = SUCCESS,
	.name          = "ndim_interface"
};

static void register_data(void)
{
	/* Initializing data */
	int val = 0;
	int i, j, k, l;
	for (l = 0; l < NT; l++)
		for (k = 0; k < NZ; k++)
			for (j = 0; j < NY; j++)
				for (i = 0; i < NX; i++)
					_arr4d[(l*NX*NY*NZ)+(k*NX*NY)+(j*NX)+i] = val++;

	/* Registering data */
	unsigned nn[4] = {NX, NY, NZ, NT};
	unsigned ldn[4] = {1, NX, NX*NY, NX*NY*NZ};

	starpu_ndim_data_register(&_arr4d_handle, STARPU_MAIN_RAM, (uintptr_t)_arr4d, ldn, nn, 4, sizeof(_arr4d[0]));
	starpu_ndim_data_register(&_arr4d2_handle, STARPU_MAIN_RAM, (uintptr_t)_arr4d2, ldn, nn, 4, sizeof(_arr4d2[0]));
}

static void unregister_data(void)
{
	starpu_data_unregister(_arr4d_handle);
	starpu_data_unregister(_arr4d2_handle);
}

void test_arr4d_cpu_func(void *buffers[], void *args)
{
	STARPU_SKIP_IF_VALGRIND;

	int factor = *(int*)args;
	int *nn = (int *)STARPU_NDIM_GET_NN(buffers[0]);
	unsigned *ldn = STARPU_NDIM_GET_LDN(buffers[0]);
	int nx = nn[0];
	int ny = nn[1];
	int nz = nn[2];
	int nt = nn[3];
	unsigned ldy = ldn[1];
	unsigned ldz = ldn[2];
	unsigned ldt = ldn[3];
	int *arr4d = (int *) STARPU_NDIM_GET_PTR(buffers[0]);
	int i, j, k, l;
	int val = 0;
	arr4d_config.copy_failed = SUCCESS;
	for (l = 0; l < nt; l++)
	{
		for (k = 0; k < nz; k++)
		{
			for (j = 0; j < ny; j++)
			{
				for (i = 0; i < nx; i++)
				{
					if (arr4d[(l*ldt)+(k*ldz)+(j*ldy)+i] != factor * val)
					{
						arr4d_config.copy_failed = FAILURE;
						return;
					}
					else
					{
						arr4d[(l*ldt)+(k*ldz)+(j*ldy)+i] *= -1;
						val++;
					}
				}
			}
		}
	}
}

int main(int argc, char **argv)
{
	struct data_interface_test_summary summary;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.ncuda = 2;
	conf.nopencl = 1;

	int ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	register_data();

	ret = run_tests(&arr4d_config, &summary);

	unregister_data();

	starpu_shutdown();

	if (ret) data_interface_test_summary_print(stderr, &summary);

	return data_interface_test_summary_success(&summary);
}

