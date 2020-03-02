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
#include "../test_interfaces.h"
#include "../../../helper.h"

#define NX 16
#define NY NX
#define NZ NX

/* Prototypes */
static void register_data(void);
static void unregister_data(void);
void test_block_cpu_func(void *buffers[], void *args);
#ifdef STARPU_USE_CUDA
extern void test_block_cuda_func(void *buffers[], void *_args);
#endif
#ifdef STARPU_USE_OPENCL
extern void test_block_opencl_func(void *buffers[], void *args);
#endif

static starpu_data_handle_t _block_handle;
static starpu_data_handle_t _block2_handle;

struct test_config block_config =
{
	.cpu_func      = test_block_cpu_func,
#ifdef STARPU_USE_CUDA
	.cuda_func     = test_block_cuda_func,
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_func   = test_block_opencl_func,
#endif
#ifdef STARPU_USE_MIC
	.cpu_func_name = "test_block_cpu_func",
#endif
	.handle        = &_block_handle,
	.dummy_handle  = &_block2_handle,
	.copy_failed   = SUCCESS,
	.name          = "block_interface"
};

static int _block[NX*NY*NZ];
static int _block2[NX*NY*NZ];

static void
register_data(void)
{
	/* Initializing data */
	int val = 0;
	int i, j, k;
	for (k = 0; k < NZ; k++)
		for (j = 0; j < NY; j++)
			for (i = 0; i < NX; i++)
                                _block[(k*NX*NY)+(j*NX)+i] = val++;

	/* Registering data */
	starpu_block_data_register(&_block_handle,
                                    STARPU_MAIN_RAM,
                                    (uintptr_t)_block,
				    NX,
				    NX * NY,
				    NX,
				    NY,
				    NZ,
				    sizeof(_block[0]));
	starpu_block_data_register(&_block2_handle,
                                    STARPU_MAIN_RAM,
                                    (uintptr_t)_block2,
				    NX,
				    NX * NY,
				    NX,
				    NY,
				    NZ,
				    sizeof(_block2[0]));
}

static void
unregister_data(void)
{
	starpu_data_unregister(_block_handle);
	starpu_data_unregister(_block2_handle);
}

void test_block_cpu_func(void *buffers[], void *args)
{
	STARPU_SKIP_IF_VALGRIND;

	int factor = *(int*)args;
	int nx = STARPU_BLOCK_GET_NX(buffers[0]);
	int ny = STARPU_BLOCK_GET_NY(buffers[0]);
	int nz = STARPU_BLOCK_GET_NZ(buffers[0]);
        unsigned ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
        unsigned ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
	int *block = (int *) STARPU_BLOCK_GET_PTR(buffers[0]);
	int i, j, k;
	int val = 0;
	block_config.copy_failed = SUCCESS;
	for (k = 0; k < nz; k++)
	{
		for (j = 0; j < ny; j++)
		{
			for (i = 0; i < nx; i++)
			{
                                if (block[(k*ldz)+(j*ldy)+i] != factor * val)
				{
					block_config.copy_failed = FAILURE;
					return;
				}
				else
				{
					block[(k*ldz)+(j*ldy)+i] *= -1;
					val++;
				}
			}
		}
	}
}

int
main(int argc, char **argv)
{
	struct data_interface_test_summary summary;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.ncuda = 2;
	conf.nopencl = 1;
	conf.nmic = -1;

	if (starpu_initialize(&conf, &argc, &argv) == -ENODEV || starpu_cpu_worker_get_count() == 0)
		goto enodev;

	register_data();

	run_tests(&block_config, &summary);

	unregister_data();

	starpu_shutdown();

	data_interface_test_summary_print(stderr, &summary);

	return data_interface_test_summary_success(&summary);

enodev:
	return STARPU_TEST_SKIPPED;
}

